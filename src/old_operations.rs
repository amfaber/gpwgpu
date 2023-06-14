pub struct SeparableConvolution<const N: usize> {
    input_pass: FullComputePass,
    passes: [FullComputePass; 2],
}

impl<const N: usize> SeparableConvolution<N> {
    pub fn new<'a>(
        device: &wgpu::Device,
        shader_specs: ShaderSpecs,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        temp_buffer: &wgpu::Buffer,
        additional_buffers: impl IntoIterator<Item = &'a wgpu::Buffer>,
    ) -> Result<Self, ProcessShaderError>{
        let shader = PREPROCESSOR.process_by_key("separable_convolution", shader_specs)?;
        let pipeline = shader.build(device);
        Ok(Self::custom(
            device,
            pipeline,
            input_buffer,
            output_buffer,
            temp_buffer,
            additional_buffers,
        ))
    }
    
    pub fn custom<'a>(
        device: &wgpu::Device,
        pipeline: Rc<NonBoundPipeline>,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        temp_buffer: &wgpu::Buffer,
        additional_buffers: impl IntoIterator<Item = &'a wgpu::Buffer>,
    ) -> Self {
        let mut bind_group_entries_output_first = vec![
            (0, output_buffer),
            (1, temp_buffer),
        ];

        let mut bind_group_entries_temp_first = vec![
            (0, temp_buffer),
            (1, output_buffer),
        ];

        let first_out = if N % 2 == 0 {
            temp_buffer
        } else {
            output_buffer
        };

        let mut bind_group_entries_input = vec![
            (0, input_buffer),
            (1, first_out),
        ];

        for (i, additional_buffer) in additional_buffers.into_iter().enumerate() {
            bind_group_entries_output_first.push((i as u32 + 2, additional_buffer));
            bind_group_entries_temp_first.push((i as u32 + 2, additional_buffer));
            bind_group_entries_input.push((i as u32 + 2, additional_buffer));
        }

        let bind_group_output_first = bind_group_entries_output_first.binding_group(device, &pipeline.bind_group_layout, None);
			
        let bind_group_temp_first = bind_group_entries_temp_first.binding_group(device, &pipeline.bind_group_layout, None);

        let bind_group_input = bind_group_entries_input.binding_group(device, &pipeline.bind_group_layout, None);

        let full_output_first = FullComputePass {
            pipeline: Rc::clone(&pipeline),
            bindgroup: bind_group_output_first,
        };

        let full_temp_first = FullComputePass {
            pipeline: Rc::clone(&pipeline),
            bindgroup: bind_group_temp_first,
        };

        let full_input = FullComputePass {
            pipeline: Rc::clone(&pipeline),
            bindgroup: bind_group_input,
        };

        let base = if N % 2 == 0 {
            [full_temp_first, full_output_first]
        } else {
            [full_output_first, full_temp_first]
        };

        Self {
            passes: base,
            input_pass: full_input,
        }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, push_constants: &[u8]) {
        let input = std::iter::once(&self.input_pass);
        for (dim, pass) in input
            .chain(self.passes.iter().cycle().take(N - 1))
            .enumerate()
        {
            let push_constants = Vec::from_iter(unsafe {
                any_as_u8_slice(&(dim as u32))
                    .iter()
                    .chain(push_constants.iter())
                    .cloned()
            });
            pass.execute(encoder, &push_constants[..]);
        }
    }
}

pub struct Laplace<const N: usize> {
    pass: SeparableConvolution<N>,
}

impl<const N: usize> Laplace<N> {
    pub fn new<'a>(
        device: &wgpu::Device,
        shader_specs: ShaderSpecs,
        input_buffer: &wgpu::Buffer,
        output_buffer: &'a wgpu::Buffer,
        temp1: &wgpu::Buffer,
        temp2: &wgpu::Buffer,
        additional_buffers: impl IntoIterator<Item = &'a wgpu::Buffer>,
    ) -> Result<Self, ProcessShaderError>{
        let shader = PREPROCESSOR.process_by_key("laplace", shader_specs)?;
        let pipeline = shader.build(device);
        Ok(Self::custom(
            device,
            pipeline,
            input_buffer,
            output_buffer,
            temp1,
            temp2,
            additional_buffers,
        ))
    }
    
    pub fn custom<'a>(
        device: &wgpu::Device,
        pipeline: Rc<NonBoundPipeline>,
        input_buffer: &wgpu::Buffer,
        output_buffer: &'a wgpu::Buffer,
        temp1: &wgpu::Buffer,
        temp2: &wgpu::Buffer,
        additional_buffers: impl IntoIterator<Item = &'a wgpu::Buffer>,
    ) -> Self {
        let additional_buffers = [output_buffer]
            .into_iter()
            .chain(additional_buffers.into_iter());
        let sep = SeparableConvolution::<N>::custom(
            device,
            pipeline,
            input_buffer,
            temp1,
            temp2,
            additional_buffers,
        );

        Self { pass: sep }
    }

    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, push_constants: &[u8]) {
        for diff_dim in 0..(N as u32) {
            let push_constants = Vec::from_iter(unsafe {
                any_as_u8_slice(&(diff_dim))
                    .iter()
                    .chain(push_constants.iter())
                    .cloned()
            });
            self.pass.execute(encoder, &push_constants[..]);
        }
    }
}
