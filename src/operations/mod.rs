use gpwgpu_macros::parse_shaders_crate;

parse_shaders_crate!{pub PREPROCESSOR, "src/operations/shaders"}

pub mod convolutions;
pub mod reductions;
pub mod simple;
