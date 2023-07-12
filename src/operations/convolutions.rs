use std::rc::Rc;

use my_core::{shaderpreprocessor::ShaderSpecs, parser::{Definition, ExpansionError}, utils::{FullComputePass, Dispatcher}};

use super::PREPROCESSOR;

pub struct GaussianSmoothing<const N: usize> {
    input_pass: FullComputePass,
    temp_pass: [FullComputePass; 2],
    // output_pass: FullComputePass,
    dims: [u32; N],
}

impl<const N: usize> GaussianSmoothing<N> {
    pub fn new<'proc>(
        device: &wgpu::Device,
        image_dims: [u32; N],
        input: &wgpu::Buffer,
        temp: &wgpu::Buffer,
        output: &wgpu::Buffer,
        sigma: f32,
    ) -> Result<Self, ExpansionError<'proc>> {
        let sigma2 = sigma.powi(2);
        let kernel_func = format!("exp(-pow(f32(#I), 2.0) / (2.0 * {sigma2:.1}))");
        let specs = ShaderSpecs::new((256, 1, 1))
            .extend_defs([
                ("N".into(), Definition::Int(N as i32)),
                // ("LOCALSIZE".into(), Definition::Int(500)),
                ("RINT".into(), Definition::UInt(5)),
                ("KERNEL_FUNC".into(), Definition::Any(kernel_func.into())),
                ("EXTRA_BUFFERS".into(), Definition::default()),
                ("NORMALIZE".into(), Definition::Bool(true)),
                ("EXTRA_PUSHCONSTANTS".into(), Definition::default()),
                ("BOUNDARY".into(), Definition::default()),
            ])
            .direct_dispatcher(&[image_dims.iter().product::<u32>(), 1, 1]);
        
        let shader = PREPROCESSOR.process_by_name("1d_strides", specs).unwrap();
        // println!("{}", shader.source);

        let pipeline = shader.build(device);

        let input_pass = {
            let first_pass_output = if N % 2 == 0{
                temp
            } else {
                output
            };
            let bindgroup = [(0, input), (1, first_pass_output)];
            FullComputePass::new(device, Rc::clone(&pipeline), &bindgroup)
        };

        let temp_pass = {
            let bindgroup = [(0, temp), (1, output)];
            FullComputePass::new(device, Rc::clone(&pipeline), &bindgroup)
        };

        let output_pass = {
            let bindgroup = [(0, output), (1, temp)];
            FullComputePass::new(device, Rc::clone(&pipeline), &bindgroup)
        };

        let temp_pass = if N % 2 == 0{
            [temp_pass, output_pass]
        } else {
            [output_pass, temp_pass]
        };

        Ok(Self{
            input_pass,
            temp_pass,
            dims: image_dims,
        })

    }

    fn rotate_dims(dims: &mut [u32; N]){
        let old_dims = dims.clone();
        dims[0] = old_dims[N-1];
        for i in 1..N{
            dims[i] = old_dims[i - 1];
        }
        dbg!(dims);
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ){
        let mut dims = self.dims.clone();
        self.input_pass.execute(encoder, bytemuck::cast_slice(&dims));
        Self::rotate_dims(&mut dims);
        for i in 0..N-1{
            let pass = &self.temp_pass[i % 2];
            pass.execute(encoder, bytemuck::cast_slice(&dims));
            Self::rotate_dims(&mut dims);
        }
    }
}
