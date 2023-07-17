// use crate::shaderpreprocessor::ShaderProcessor;
use macros::parse_shaders_crate;
// use once_cell::sync::Lazy;

// pub static PREPROCESSOR: Lazy<ShaderProcessor> = 
//     Lazy::new(|| parse_shaders_crate!("src/operations/shaders"));

parse_shaders_crate!{pub PREPROCESSOR, "src/operations/shaders"}

pub mod convolutions;
pub mod reductions;
