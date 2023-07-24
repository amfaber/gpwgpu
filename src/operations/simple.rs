use std::borrow::Cow;

use my_core::{utils::FullComputePass, parser::{ExpansionError, Definition}, shaderpreprocessor::ShaderSpecs};

use super::PREPROCESSOR;

pub enum OperationType<'a>{
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Custom(Cow<'a, str>),
}

impl<'a> OperationType<'a>{
    fn to_def(&self, unary_binary: &UnaryBinary) -> Definition{
        let other = match unary_binary{
            UnaryBinary::Unary(constant) => format!("{:.1}", constant),
            UnaryBinary::Binary(_) => "input2[idx]".to_string(),
        };
        use OperationType::*;
        match self{
            Add => Definition::Any(format!("input[idx] + {other}").into()),
            Sub => Definition::Any(format!("input[idx] - {other}").into()),
            Mul => Definition::Any(format!("input[idx] * {other}").into()),
            Div => Definition::Any(format!("input[idx] / {other}").into()),
            Max => Definition::Any(format!("max(input[idx], {other})").into()),
            Min => Definition::Any(format!("min(input[idx], {other})").into()),
            Custom(cow) => Definition::Any(cow.clone()),
        }
    }
}

enum UnaryBinary<'a>{
    Unary(f32),
    Binary(&'a wgpu::Buffer)
}

fn new_simple(
    device: &wgpu::Device,
    length: usize,
    ty: OperationType,
    input: &wgpu::Buffer,
    unary_binary: UnaryBinary,
    output: Option<&wgpu::Buffer>,
) -> Result<FullComputePass, ExpansionError>{
    
    let specs = ShaderSpecs::new((256, 1, 1))
        .direct_dispatcher(&[length as u32, 1, 1])
        .extend_defs([
            ("LENGTH", (length as u32).into()),
            ("BINARY", true.into()),
            ("INPLACE", false.into()),
            ("OPERATION", ty.to_def(&unary_binary)),
        ]);

    let shader = PREPROCESSOR.process_by_name("simple", specs)?;

    let pipeline = shader.build(device);

    let mut bindgroup = vec![
        (0, input),
    ];

    if let UnaryBinary::Binary(binary) = unary_binary{
        bindgroup.push((1, binary));
    }

    if let Some(output) = output{
        bindgroup.push((2, output));
    }

    let pass = FullComputePass::new(device, pipeline, &bindgroup);
    
    Ok(pass)
}

pub struct BinaryOutplace(FullComputePass);

impl BinaryOutplace{
    pub fn new(
        device: &wgpu::Device,
        length: usize,
        ty: OperationType,
        input: &wgpu::Buffer,
        input2: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> Result<Self, ExpansionError>{
        Ok(Self(new_simple(
            device,
            length,
            ty,
            input,
            UnaryBinary::Binary(input2),
            Some(output),
        )?))
    }
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder){
        self.0.execute(encoder, &[]);
    }
}

pub struct BinaryInplace(FullComputePass);

impl BinaryInplace{
    pub fn new(
        device: &wgpu::Device,
        length: usize,
        ty: OperationType,
        input: &wgpu::Buffer,
        input2: &wgpu::Buffer,
    ) -> Result<Self, ExpansionError>{
        Ok(Self(new_simple(
            device,
            length,
            ty,
            input,
            UnaryBinary::Binary(input2),
            None,
        )?))
    }
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder){
        self.0.execute(encoder, &[]);
    }
}

pub struct UnaryInplace(FullComputePass);

impl UnaryInplace{
    pub fn new(
        device: &wgpu::Device,
        length: usize,
        ty: OperationType,
        input: &wgpu::Buffer,
        constant: f32,
    ) -> Result<Self, ExpansionError>{
        Ok(Self(new_simple(
            device,
            length,
            ty,
            input,
            UnaryBinary::Unary(constant),
            None,
        )?))
    }
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder){
        self.0.execute(encoder, &[]);
    }
}

pub struct UnaryOutplace(FullComputePass);

impl UnaryOutplace{
    pub fn new(
        device: &wgpu::Device,
        length: usize,
        ty: OperationType,
        input: &wgpu::Buffer,
        constant: f32,
        output: &wgpu::Buffer,
    ) -> Result<Self, ExpansionError>{
        Ok(Self(new_simple(
            device,
            length,
            ty,
            input,
            UnaryBinary::Unary(constant),
            Some(output),
        )?))
    }
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder){
        self.0.execute(encoder, &[]);
    }
}