use std::rc::Rc;

#[allow(unused)]
use my_core::shaderpreprocessor::ShaderProcessor;

use my_core::{shaderpreprocessor::{ShaderSpecs, NonBoundPipeline}, parser::{Definition, ExpansionError}, utils::{FullComputePass, any_as_u8_slice}};

use super::PREPROCESSOR;

pub fn rotate_dims<const N: usize>(dims: &mut [i32; N]){
    let old_dims = dims.clone();
    dims[0] = old_dims[N-1];
    for i in 1..N{
        dims[i] = old_dims[i - 1];
    }
}

pub struct SeparableConvolution<const N: usize>{
    input_pass: FullComputePass,
    temp_pass: [FullComputePass; 2],
    dims: [i32; N],
}

impl<const N: usize> SeparableConvolution<N>{
    pub fn from_two_pipelines<'buf, 'proc>(
        device: &wgpu::Device,
        first_pipeline: Rc<NonBoundPipeline>,
        pipeline: Rc<NonBoundPipeline>,
        dims: [i32; N],
        input: &wgpu::Buffer,
        temp: &wgpu::Buffer,
        output: &wgpu::Buffer,
        first_additional_buffers: impl IntoIterator<Item = &'buf wgpu::Buffer> + Copy,
        additional_buffers: impl IntoIterator<Item = &'buf wgpu::Buffer> + Copy,
    ) -> Result<Self, ExpansionError> {
        
        let input_pass = {
            let first_pass_output = if N % 2 == 0{
                temp
            } else {
                output
            };
            let mut bindgroup = vec![(0, input), (1, first_pass_output)];
            bindgroup.extend(first_additional_buffers.into_iter()
                .enumerate()
                .map(|(i, buffer)| ((i + 2) as u32, buffer))
            );
            FullComputePass::new(device, Rc::clone(&first_pipeline), &bindgroup)
        };

        let temp_pass = {
            let mut bindgroup = vec![(0, temp), (1, output)];
            bindgroup.extend(additional_buffers.into_iter()
                .enumerate()
                .map(|(i, buffer)| ((i + 2) as u32, buffer))
            );
            FullComputePass::new(device, Rc::clone(&pipeline), &bindgroup)
        };

        let output_pass = {
            let mut bindgroup = vec![(0, output), (1, temp)];
            bindgroup.extend(additional_buffers.into_iter()
                .enumerate()
                .map(|(i, buffer)| ((i + 2) as u32, buffer))
            );
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
            dims,
        })
    }
    
    pub fn from_pipeline<'buf, 'proc>(
        device: &wgpu::Device,
        pipeline: Rc<NonBoundPipeline>,
        dims: [i32; N],
        input: &wgpu::Buffer,
        temp: &wgpu::Buffer,
        output: &wgpu::Buffer,
        additional_buffers: impl IntoIterator<Item = &'buf wgpu::Buffer> + Copy,
    ) -> Result<Self, ExpansionError> {
        Self::from_two_pipelines(
            device,
            Rc::clone(&pipeline),
            pipeline,
            dims,
            input,
            temp,
            output,
            additional_buffers,
            additional_buffers,
        )
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        extra_push_constants: &[u8],
    ){
        self.execute_many_push(encoder, [extra_push_constants; N])
    }

    pub fn execute_many_push(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        extra_push_constants: [&[u8]; N],
    ){
        let mut dims = self.dims.clone();
        let mut push = bytemuck::cast_slice(&dims).to_vec();
        push.extend_from_slice(extra_push_constants[0]);

        self.input_pass.execute(encoder, &push);
        rotate_dims(&mut dims);
        for i in 0..N-1{
            let pass = &self.temp_pass[i % 2];
            let mut push = bytemuck::cast_slice(&dims).to_vec();
            push.extend_from_slice(extra_push_constants[i + 1]);
            pass.execute(encoder, &push);
            rotate_dims(&mut dims);
        }
    }
}


pub struct GaussianSmoothing<const N: usize>(SeparableConvolution<N>);

impl<const N: usize> GaussianSmoothing<N> {
    pub fn pipeline(
        device: &wgpu::Device,
        dims: [i32; N],
    ) -> Result<Rc<NonBoundPipeline>, ExpansionError> {
        let extra_pushconstants = "sigma: f32,";
        let init = "let sigma2 = pow(pc.sigma, 2.0);";
        let kernel_func = "\
        kernel_eval = exp(-pow(f32(i), 2.0) / (2.0 * sigma2));
        acc += input[idx + i] * kernel_eval;";
        let rint_expr = "i32(4.0 * pc.sigma + 0.5)";
        let output_str = "output[idx] = acc;";
        let specs = ShaderSpecs::new((256, 1, 1))
            .extend_defs([
                ("N", (N as i32).into()),
                // ("LOCALSIZE".into(), Definition::Int(500)),

                ("RINT_EXPR", rint_expr.into()),
                ("KERNEL_FUNC", kernel_func.into()),
                ("EXTRA_BUFFERS", "".into()),
                ("NORMALIZE", true.into()),
                ("EXTRA_PUSHCONSTANTS", extra_pushconstants.into()),
                ("BOUNDARY", "".into()),
                ("OUTPUT", output_str.into()),
                ("INIT", init.into()),
                ("POST", "".into()),
            ])
            .direct_dispatcher(&[dims.iter().map(|&x| x as u32).product::<u32>(), 1, 1]);
        
        let shader = PREPROCESSOR.process_by_name("1d_strides", specs)?;

        Ok(shader.build(device))
    }

    
    pub fn from_pipeline(
        device: &wgpu::Device,
        dims: [i32; N],
        pipeline: Rc<NonBoundPipeline>,
        input: &wgpu::Buffer,
        temp: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<Self, ExpansionError> {
        let pass = SeparableConvolution::from_pipeline(
            device,
            pipeline,
            dims,
            input,
            temp,
            output,
            None,
        )?;

        Ok(Self(pass))
    }

    
    pub fn new(
        device: &wgpu::Device,
        dims: [i32; N],
        input: &wgpu::Buffer,
        temp: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<Self, ExpansionError> {
        let pipeline = Self::pipeline(
            device,
            dims,
        )?;
        let pass = SeparableConvolution::from_pipeline(
            device,
            pipeline,
            dims,
            input,
            temp,
            output,
            None,
        )?;
        Ok(Self(pass))
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        sigma: [f32; N],
    ){
        let push = sigma.iter().map(|s| unsafe{ any_as_u8_slice(s) }).collect::<Vec<_>>();
        self.0.execute_many_push(encoder, push.try_into().unwrap());
    }
}


#[derive(Clone, Debug, Copy)]
#[repr(C)]
struct GaussianLaplacePush{
    sigma: f32,
    diff: i32,
    last: i32,
}

pub struct GaussianLaplace<'a, const N: usize>{
    pass: SeparableConvolution<N>,
    output: &'a wgpu::Buffer,
}

impl<'a, const N: usize> GaussianLaplace<'a, N>{
    pub fn pipeline(
        device: &wgpu::Device,
        dims: [i32; N],
    ) -> Result<Rc<NonBoundPipeline>, ExpansionError> {
        
        let extra_pushconstants = "sigma: f32,\ndiff: i32,\nlast: i32";
        let init = "let sigma2 = pow(pc.sigma, 2.0);";

        let kernel_func = "\
        let x2 = pow(f32(i), 2.0);
        if pc.diff == 1{
            kernel_eval = exp(-x2 / (2.0 * sigma2)) * (x2 - sigma2);
        } else{
            kernel_eval = exp(-x2 / (2.0 * sigma2));
        }
        acc += input[idx + i] * kernel_eval;";

        let rint_expr = "i32(4.0 * pc.sigma + 0.5)";

        let extra_buffers = "\
@group(0) @binding(2)
var<storage, read_write> final_output: array<f32>;";

        let output_str = "\
    if pc.last == 1{
        final_output[idx] += acc;
    } else {
        output[idx] = acc;
    }";
        let specs = ShaderSpecs::new((256, 1, 1))
            .extend_defs([
                ("N", Definition::Int(N as i32)),
                // ("LOCALSIZE", 500.into()),

                ("RINT_EXPR", rint_expr.into()),
                ("KERNEL_FUNC", kernel_func.into()),
                ("EXTRA_BUFFERS", "".into()),
                ("NORMALIZE", true.into()),
                ("EXTRA_PUSHCONSTANTS", extra_pushconstants.into()),
                ("BOUNDARY", "".into()),
                ("EXTRA_BUFFERS", extra_buffers.into()),
                ("INIT", init.into()),
                ("POST", "".into()),
                ("OUTPUT", output_str.into()),
            ])
            .direct_dispatcher(&[dims.iter().map(|&x| x as u32).product::<u32>(), 1, 1]);
        
        let shader = PREPROCESSOR.process_by_name("1d_strides", specs)?;

        Ok(shader.build(device))

    }
    
    pub fn from_pipeline(
        device: &wgpu::Device,
        dims: [i32; N],
        pipeline: Rc<NonBoundPipeline>,
        input: &wgpu::Buffer,
        temp: &wgpu::Buffer,
        temp2: &wgpu::Buffer,
        output: &'a wgpu::Buffer,
    ) -> Result<Self, ExpansionError> {
        let pass = SeparableConvolution::from_pipeline(
            device,
            pipeline,
            dims,
            input,
            temp,
            temp2,
            Some(output)
        )?;

        Ok(Self{
            pass,
            output,
        })
    }

    pub fn new(
        device: &wgpu::Device,
        dims: [i32; N],
        input: &wgpu::Buffer,
        temp: &wgpu::Buffer,
        temp2: &wgpu::Buffer,
        output: &'a wgpu::Buffer,
    ) -> Result<Self, ExpansionError> {
        let pipeline = Self::pipeline(device, dims)?;
        Self::from_pipeline(
            device,
            dims,
            pipeline,
            input,
            temp,
            temp2,
            output,
        )
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        sigma: [f32; N],
    ){
        encoder.clear_buffer(self.output, 0, None);
        for i in 0..N{
            let push = GaussianLaplacePush{
                sigma: sigma[i],
                diff: 0,
                last: 0,
            };
            let mut push = [push; N];
            push[N - 1].last = 1;
            push[i].diff = 1;
            let push_u8 = push.iter().map(|x| unsafe{ any_as_u8_slice(x) }).collect::<Vec<_>>();
            self.pass.execute_many_push(encoder, push_u8.try_into().unwrap());
        }
    }
}


