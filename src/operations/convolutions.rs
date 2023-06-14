use my_core::shaderpreprocessor::{ShaderSpecs, ShaderDefVal};

use super::PREPROCESSOR;


pub struct GaussianSmoothing{
    
}


impl GaussianSmoothing{
    pub fn new<const N: usize>(
        device: &wgpu::Device,
        image_dims: &[u32; N],
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        sigma: f32,
    ){

        let specs = ShaderSpecs::new( (256, 1, 1) )
            .extend_defs(&[
                ShaderDefVal::UInt("NDIM".into(), N as u32),
                ShaderDefVal::Int("LOCALSIZE".into(), 500),
                ShaderDefVal::Int("image_dims_i".into(), image_dims[0] as i32),
                ShaderDefVal::Int("image_dims_j".into(), image_dims[1] as i32),
                ShaderDefVal::Int("image_dims_k".into(), image_dims[2] as i32),
            ]);
        let shader =PREPROCESSOR.process_by_key("coalesced_separable_convolution", specs).unwrap();
        print!("{}", shader.source);
        shader.build(device);
    }
    
}
