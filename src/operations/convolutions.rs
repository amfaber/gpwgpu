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
                ShaderDefVal::UInt("N".into(), N as u32),
                ShaderDefVal::UInt("LOCALSIZE".into(), 500),
            ]);
        let shader = PREPROCESSOR.process_by_key("1d_strides", specs).unwrap();

        print!("{}", shader.source);
        shader.build(device);
    }
    
}
