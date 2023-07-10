use my_core::shaderpreprocessor::{ShaderDefVal, ShaderSpecs};

use super::PREPROCESSOR;

pub struct GaussianSmoothing {}

impl GaussianSmoothing {
    pub fn new<const N: usize>(
        device: &wgpu::Device,
        image_dims: &[u32; N],
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        sigma: f32,
    ) {
        let specs = ShaderSpecs::new((256, 1, 1)).extend_defs(&[
            ShaderDefVal::Int("N".into(), N as i32),
            ShaderDefVal::UInt("LOCALSIZE".into(), 500),
            ShaderDefVal::UInt("RINT".into(), 5),
        ]);
        let shader = PREPROCESSOR.process_by_key("1d_strides", specs).unwrap();

        print!("{}", shader.source);
        shader.build(device);
    }
}
