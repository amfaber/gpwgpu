use crate::shaderpreprocessor::*;
use crate::utils::*;
use my_core::parser::Definition;
use my_core::parser::ExpansionError;
use wgpu::CommandEncoder;

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
    fn to_shader_val(&self) -> (String, Definition) {
        let op_str = "OPERATION".to_string();
        match self {
            Self::Sum => (op_str, Definition::Any("acc += datum;".into())),
            Self::Product => (op_str, Definition::Any("acc *= datum;".into())),
            Self::Min => (op_str, Definition::Any("acc = min(acc, datum);".into())),
            Self::Max => (op_str, Definition::Any("acc = max(acc, datum);".into())),
            Self::Custom(operation) => (op_str, Definition::Any(operation.into())),
        }
    }
}

pub struct Reduce {
    outplace_pass: Option<FullComputePass>,
    inplace_pass: FullComputePass,

    last_reduction: FullComputePass,
    size: usize,
    workgroup_size: WorkgroupSize,
    unroll: u32,
    last_size: u32,
}

impl Reduce {
    pub fn new<'buf, 'proc>(
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        temp: Option<&wgpu::Buffer>,
        output: &wgpu::Buffer,
        size: usize,
        ty: ReductionType,
        nanprotection: bool,
        unroll: u32,
        specs: ShaderSpecs,

        extra_push: String,
        extra_last: String,
        extra_buffers: String,
        last_buffers: impl IntoIterator<Item = &'buf wgpu::Buffer>,

        last_size: u32,

        inplace_label: &str,
    ) -> Result<Self, ExpansionError<'proc>> {
        let workgroup_size = specs.workgroup_size.clone();

        let outplace_pass = match temp {
            Some(output) => {
                let outplace_specs = specs.clone();
                let outplace_specs = outplace_specs
                    .extend_defs([
                        ("NANPROTECT".into(), Definition::Bool(nanprotection)),
                        ("OUTPLACE".into(), Definition::Bool(true)),
                        ("UNROLL".into(), Definition::UInt(unroll)),
                        ty.to_shader_val(),
                    ])
                    .labels("outplace");
                let outplace = PREPROCESSOR
                    .process_by_name("reduce", outplace_specs)?
                    .build(device);

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
                    ty.to_shader_val(),
                ])
                .labels(inplace_label);
            let inplace = PREPROCESSOR.process_by_name("reduce", specs)?;
            // print!("{}", inplace.source);
            let inplace = inplace.build(device);

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
                    ("EXTRAPUSHCONSTANTS".into(), Definition::Any(extra_push.into())),
                    ("EXTRABUFFERS".into(), Definition::Any(extra_buffers.into())),
                    ("EXTRALAST".into(), Definition::Any(extra_last.into())),
                    ty.to_shader_val(),
                ])
                .labels("last_reduction");
            let last_reduction = PREPROCESSOR.process_by_name("last_reduction", specs)?;
            let last_reduction = last_reduction.build(device);

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
            size,

            workgroup_size,

            unroll,

            last_size,
        })
    }

    fn interal_execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass: &FullComputePass,
        length: &mut u32,
    ) {
        let to_start = (*length + self.unroll - 1) / self.unroll;
        let dispatcher = Dispatcher::new_direct(&[*length, 1, 1], &self.workgroup_size);
        pass.execute_with_dispatcher(
            encoder,
            unsafe { any_as_u8_slice(&(to_start, *length)) },
            &dispatcher,
        );
        *length = to_start;
    }

    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        last_reduction_additional_pushconstants: &[u8],
    ) {
        let mut length = self.size as u32;

        if let Some(outplace_pass) = &self.outplace_pass {
            self.interal_execute(encoder, outplace_pass, &mut length)
        }

        // let mut it = 0;
        while length > self.last_size {
            self.interal_execute(encoder, &self.inplace_pass, &mut length);
            // if it == 2{
            //     return ;
            // }
            // it += 1;
        }
        let mut last_pushconstants = unsafe { any_as_u8_slice(&length) }.to_vec();
        last_pushconstants.extend_from_slice(last_reduction_additional_pushconstants);
        let dispatcher = Dispatcher::Direct([1, 1, 1]);
        self.last_reduction
            .execute_with_dispatcher(encoder, &last_pushconstants, &dispatcher);
    }
}

struct MeanReduce {
    reduction: Reduce,
}

impl MeanReduce {
    pub fn new<'proc>(
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        temp: Option<&wgpu::Buffer>,
        divisor: Option<&wgpu::Buffer>,
        output: &wgpu::Buffer,
        size: usize,
        unroll: u32,
        specs: ShaderSpecs,
        last_size: u32,
    ) -> Result<Self, ExpansionError<'proc>> {
        let extra_buffers = match divisor {
            Some(_) => "\
@group(0) @binding(2)
var<storage, read> mean_divisor: u32;"
                .to_string(),
            None => "".to_string(),
        };

        let extra_push = "".to_string();

        let extra_last = match divisor {
            Some(_) => "acc /= f32(mean_divisor);".to_string(),
            None => format!("acc /= {}.0;", size),
        };

        let last_buffers = divisor;

        let nanprotection = divisor.is_some();

        let reduction = Reduce::new(
            device,
            input,
            temp,
            output,
            size,
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
        )?;

        Ok(Self { reduction })
    }

    pub fn execute(&self, encoder: &mut CommandEncoder) {
        self.reduction.execute(encoder, &[]);
    }
}

pub struct StandardDeviationReduce {
    mean: MeanReduce,
    square_residuals: FullComputePass,
    mean_and_sqrt: Reduce,
}

impl StandardDeviationReduce {
    pub fn new<'proc>(
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        temp: &wgpu::Buffer,
        divisor: Option<&wgpu::Buffer>,
        output: &wgpu::Buffer,
        size: usize,
        unroll: u32,
        specs: ShaderSpecs,
        last_size: u32,
    ) -> Result<Self, ExpansionError<'proc>> {
        let mean = {
            MeanReduce::new(
                device,
                input,
                Some(temp),
                divisor,
                output,
                size,
                unroll,
                specs.clone(),
                last_size,
            )?
        };

        let square_residuals = {
            let wg_size = specs.workgroup_size.clone();
            let specs = ShaderSpecs::new(wg_size)
                .direct_dispatcher(&[size as u32, 1, 1])
                .extend_defs([
                    ("NANPROTECT".into(), Definition::Bool(divisor.is_some())),
                    ("TOTALELEMENTS".into(), Definition::UInt(size as u32)),
                ]);
            let shader = PREPROCESSOR
                .process_by_name("square_residuals", specs)?
                .build(device);
            let bindgroup = [(0, input), (1, temp), (2, output)];
            FullComputePass::new(device, shader, &bindgroup)
        };

        let mean_and_sqrt = {
            let extra_buffers = match divisor {
                Some(_) => "\
@group(0) @binding(2)
var<storage, read> mean_divisor: u32;"
                    .to_string(),
                None => "".to_string(),
            };

            let extra_push = "".to_string();

            let extra_last = match divisor {
                Some(_) => "acc /= f32(mean_divisor - 1u); acc = sqrt(acc);".to_string(),
                None => format!("acc /= f32({}u - 1u); acc = sqrt(acc);", size),
            };

            let last_buffers = divisor;

            let nanprotection = divisor.is_some();

            Reduce::new(
                device,
                temp,
                None,
                output,
                size,
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
            )?
        };

        Ok(Self {
            mean,
            square_residuals,
            mean_and_sqrt,
        })
    }

    pub fn execute(&self, encoder: &mut CommandEncoder) {
        self.mean.execute(encoder);
        self.square_residuals.execute(encoder, &[]);
        self.mean_and_sqrt.execute(encoder, &[]);
        return;
    }
}
