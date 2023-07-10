use crate::shaderpreprocessor::ShaderProcessor;
use macros::add_directory_crate;
use once_cell::sync::Lazy;

pub static PREPROCESSOR: Lazy<ShaderProcessor> =
    Lazy::new(|| add_directory_crate!("src/operations/shaders"));

pub mod convolutions;
pub mod reductions;
