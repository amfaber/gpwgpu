use crate::shaderpreprocessor::*;
use crate::utils::*;
use bytemuck::bytes_of;
use gpwgpu_core::parser::Definition;

use super::PREPROCESSOR;

#[derive(Clone)]
pub enum ReductionType {
    Sum,
    Product,
    Min,
    Max,
    Custom(String),
}


impl ReductionType {
    fn to_shader_val(&self) -> (&'static str, Definition) {
        match self {
            Self::Sum => ("OPERATION", Definition::Any("acc += datum;".into())),
            Self::Product => ("OPERATION", Definition::Any("acc *= datum;".into())),
            Self::Min => ("OPERATION", Definition::Any("acc = min(acc, datum);".into())),
            Self::Max => ("OPERATION", Definition::Any("acc = max(acc, datum);".into())),
            Self::Custom(operation) => ("OPERATION", Definition::Any(operation.into())),
        }
    }
}

#[derive(Clone, Default)]
pub enum InputType{
    #[default]
    F32,
    Vec4F32,
    Custom(String),
}

impl InputType {
    fn to_shader_val(&self) -> (&'static str, Definition) {
        match self {
            Self::F32 => ("INPUT_TYPE", Definition::Any("f32".into())),
            Self::Vec4F32 => ("INPUT_TYPE", Definition::Any("vec4<f32>".into())),
            Self::Custom(ty) => ("INPUT_TYPE", Definition::Any(ty.into())),
        }
    }
}

pub struct ReduceBuilder<'a>{
    pub input: &'a wgpu::Buffer,
    pub output: &'a wgpu::Buffer,
    pub temp: Option<&'a wgpu::Buffer>,
    pub ty: ReductionType,
    pub specs: ShaderSpecs<'static>,
    pub unroll: u32,
    pub last_size: u32,
    pub nanprotection: bool,
    pub input_type: InputType,
    pub inplace_label: String,
    pub extra_push: String,
    pub extra_last: String,
    pub extra_buffers: String,
    pub last_buffers: Vec<&'a wgpu::Buffer>,
    
}

impl<'a> ReduceBuilder<'a>{
    pub fn new(input: &'a wgpu::Buffer, output: &'a wgpu::Buffer, ty: ReductionType) -> Self{
        Self{
            input,
            output,
            temp: None,
            ty,
            specs: ShaderSpecs::new((256, 1, 1)),
            unroll: 8,
            last_size: 24,
            nanprotection: false,
            input_type: InputType::F32,
            inplace_label: String::new(),
            extra_push: String::new(),
            extra_last: String::new(),
            extra_buffers: String::new(),
            last_buffers: Vec::new(),
        }
    }

    pub fn build(self, device: &wgpu::Device) -> Result<Reduce, ShaderError>{
        Reduce::new(
            device,
            self.input,
            self.temp,
            self.output,
            self.ty,
            self.nanprotection,
            self.unroll,
            self.specs,
            self.extra_push,
            self.extra_last,
            self.extra_buffers,
            self.last_buffers,
            self.last_size,
            &self.inplace_label,
            self.input_type,
        )
    }
}

#[derive(Debug)]
pub struct Reduce {
    outplace_pass: Option<FullComputePass>,
    inplace_pass: FullComputePass,

    last_reduction: FullComputePass,
    // size: usize,
    workgroup_size: WorkgroupSize,
    unroll: u32,
    last_size: u32,
}

impl Reduce {
    pub fn new<'buf, 'proc>(
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        // FIXME annotate size of this buffer. Perhaps even check and fail if not sufficient
        temp: Option<&wgpu::Buffer>,
        output: &wgpu::Buffer,
        // size: usize,
        ty: ReductionType,
        nanprotection: bool,
        // Each thread adds this many numbers at a time
        unroll: u32,
        // This is to allow passing the workgroup size, push constant size, entry point and so on.
        specs: ShaderSpecs<'_>,

        extra_push: String,
        extra_last: String,
        extra_buffers: String,
        last_buffers: impl IntoIterator<Item = &'buf wgpu::Buffer>,

        // Below this number of remaining elements, we switch to the last reduction
        last_size: u32,

        inplace_label: &str,
        input_type: InputType,
    ) -> Result<Self, ShaderError> {
        let workgroup_size = specs.workgroup_size.clone();

        let outplace_pass = match temp {
            Some(output) => {
                let outplace_specs = specs.clone();
                let outplace_specs = outplace_specs
                    .extend_defs([
                        ("NANPROTECT", nanprotection.into()),
                        ("OUTPLACE", true.into()),
                        ("UNROLL", unroll.into()),
                        input_type.to_shader_val(),
                        ty.to_shader_val(),
                    ])
                    .labels("outplace");

                let shader = PREPROCESSOR.process_by_name("reduce", outplace_specs)?;
                let outplace = shader.build(device)?;

                let bindgroup = [(0, input), (1, output)];

                Some(FullComputePass::new(device, outplace, &bindgroup))
            }
            None => None,
        };

        let inplace_pass = {
            let specs = specs
                .extend_defs([
                    ("NANPROTECT".into(), Definition::Bool(false)),
                    ("OUTPLACE".into(), Definition::Bool(false)),
                    ("UNROLL".into(), Definition::UInt(unroll)),
                    input_type.to_shader_val(),
                    ty.to_shader_val(),
                ])
                .labels(inplace_label);
            let inplace = PREPROCESSOR.process_by_name("reduce", specs)?;
            let inplace = inplace.build(device)?;

            let bindgroup = [(
                0,
                match temp {
                    Some(temp) => temp,
                    None => input,
                },
            )];

            FullComputePass::new(device, inplace, &bindgroup)
        };

        let last_reduction = {
            let specs = ShaderSpecs::new((1, 1, 1))
                .direct_dispatcher(&[1, 1, 1])
                .extend_defs([
                    (
                        "EXTRAPUSHCONSTANTS".into(),
                        Definition::Any(extra_push.into()),
                    ),
                    ("EXTRABUFFERS".into(), Definition::Any(extra_buffers.into())),
                    ("EXTRALAST".into(), Definition::Any(extra_last.into())),
                    input_type.to_shader_val(),
                    ty.to_shader_val(),
                ])
                .labels("last_reduction");
            let last_reduction = PREPROCESSOR.process_by_name("last_reduction", specs)?;
            let last_reduction = last_reduction.build(device)?;

            let last_input = match temp {
                Some(buffer) => buffer,
                None => input,
            };

            let mut bindgroup = vec![(0, last_input), (1, output)];

            for (i, extra_buffer) in last_buffers.into_iter().enumerate() {
                bindgroup.push((i as u32 + 2, extra_buffer));
            }

            FullComputePass::new(device, last_reduction, &bindgroup)
        };

        Ok(Self {
            outplace_pass,
            inplace_pass,

            last_reduction,
            // size,

            workgroup_size,

            unroll,

            last_size,
        })
    }

    fn internal_execute(
        &self,
        encoder: &mut Encoder,
        pass: &FullComputePass,
        length: &mut u32,
    ) {
        let to_start = (*length + self.unroll - 1) / self.unroll;
        let flat_dispatcher = dispatcher_flat(to_start as u64, self.workgroup_size.clone());
        let dispatcher = Dispatcher::Direct(flat_dispatcher);
        pass.execute_with_dispatcher(
            encoder,
            bytemuck::bytes_of(&[to_start, *length]),
            &dispatcher,
        );
        *length = to_start;
    }

    pub fn execute(
        &self,
        encoder: &mut Encoder,
        mut length: u32,
        last_reduction_additional_pushconstants: &[u8],
    ) {
        // let mut length = self.size as u32;

        if let Some(outplace_pass) = &self.outplace_pass {
            self.internal_execute(encoder, outplace_pass, &mut length)
        }

        while length > self.last_size {
            self.internal_execute(encoder, &self.inplace_pass, &mut length);
        }
        let mut last_pushconstants = bytemuck::bytes_of(&length).to_vec();
        last_pushconstants.extend_from_slice(last_reduction_additional_pushconstants);
        let dispatcher = Dispatcher::Direct([1, 1, 1]);
        self.last_reduction
            .execute_with_dispatcher(encoder, &last_pushconstants, &dispatcher);
    }
}

#[derive(Debug)]
pub struct MeanReduce {
    reduction: Reduce,
}

impl MeanReduce {
    pub fn new(
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        temp: Option<&wgpu::Buffer>,
        divisor: Option<&wgpu::Buffer>,
        output: &wgpu::Buffer,
        // size: usize,
        unroll: u32,
        specs: ShaderSpecs<'_>,
        last_size: u32,
        input_type: InputType,
    ) -> Result<Self, ShaderError> {
        let extra_buffers = match divisor {
            Some(_) => "\
@group(0) @binding(2)
var<storage, read> mean_divisor: u32;"
                .to_string(),
            None => "".to_string(),
        };

        let extra_push = "length: u32,".to_string();

        let extra_last = match divisor {
            Some(_) => "acc /= f32(mean_divisor);".to_string(),
            None => {
                match input_type{
                    InputType::F32 => "acc /= f32(pc.length);".to_string(),
                    InputType::Vec4F32 => "acc /= vec4(f32(pc.length));".to_string(),
                    InputType::Custom(_) => panic!("Unsupported so far"),
                }
            },
        };

        let last_buffers = divisor;

        let nanprotection = divisor.is_some();

        let reduction = Reduce::new(
            device,
            input,
            temp,
            output,
            // size,
            ReductionType::Sum,
            nanprotection,
            unroll,
            specs,
            extra_push,
            extra_last,
            extra_buffers,
            last_buffers,
            last_size,
            "mean_inplace",
            input_type,
        )?;

        Ok(Self { reduction })
    }

    pub fn execute(&self, encoder: &mut Encoder, length: u32) {
        let push = bytes_of(&length);
        self.reduction.execute(encoder, length, push);
    }
}

#[derive(Debug)]
pub struct StandardDeviationReduce {
    mean: MeanReduce,
    square_residuals: (FullComputePass, WorkgroupSize),
    mean_and_sqrt: Reduce,
}

impl StandardDeviationReduce {
    pub fn new(
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        temp: &wgpu::Buffer,
        divisor: Option<&wgpu::Buffer>,
        output: &wgpu::Buffer,
        // size: usize,
        unroll: u32,
        specs: ShaderSpecs<'_>,
        last_size: u32,
        input_type: InputType,
    ) -> Result<Self, ShaderError> {
        let mean = {
            MeanReduce::new(
                device,
                input,
                Some(temp),
                divisor,
                output,
                // size,
                unroll,
                specs.clone(),
                last_size,
                input_type.clone(),
            )?
        };

        let square_residuals = {
            let wg_size = specs.workgroup_size.clone();
            let specs = ShaderSpecs::new(wg_size.clone())
                // .direct_dispatcher(&[size as u32, 1, 1])
                .extend_defs([
                    ("NANPROTECT", Definition::Bool(divisor.is_some())),
                    input_type.to_shader_val(),
                    // ("TOTALELEMENTS", Definition::UInt(size as u32)),
                ]);
            let shader = PREPROCESSOR
                .process_by_name("square_residuals", specs)?
                .build(device)?;
            let bindgroup = [(0, input), (1, temp), (2, output)];
            (FullComputePass::new(device, shader, &bindgroup), wg_size)
        };

        let mean_and_sqrt = {
            let extra_buffers = match divisor {
                Some(_) => "\
@group(0) @binding(2)
var<storage, read> mean_divisor: u32;"
                    .to_string(),
                None => "".to_string(),
            };

            let extra_push = "length: u32,".to_string();

            let extra_last = match divisor {
                Some(_) => "acc /= f32(mean_divisor - 1u); acc = sqrt(acc);".to_string(),
                None => "acc /= f32(pc.length - 1u); acc = sqrt(acc);".to_string(),
            };

            let last_buffers = divisor;

            let nanprotection = divisor.is_some();

            Reduce::new(
                device,
                temp,
                None,
                output,
                // size,
                ReductionType::Sum,
                nanprotection,
                unroll,
                specs,
                extra_push,
                extra_last,
                extra_buffers,
                last_buffers,
                last_size,
                "std_inplace",
                input_type,
            )?
        };

        Ok(Self {
            mean,
            square_residuals,
            mean_and_sqrt,
        })
    }

    pub fn execute(&self, encoder: &mut Encoder, length: u32) {
        self.mean.execute(encoder, length);
        let push = bytemuck::bytes_of(&length);
        let dispatcher = dispatcher_flat(length as _, self.square_residuals.1.clone());
        let square_dispatcher = Dispatcher::Direct(dispatcher);
        self.square_residuals.0.execute_with_dispatcher(encoder, push, &square_dispatcher);
        self.mean_and_sqrt.execute(encoder, length, push);
    }
}
