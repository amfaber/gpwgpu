use std::time::{Duration, Instant};

use gpwgpu::{
    operations::{
        convolutions::GaussianSmoothing,
        reductions::{Reduce, ReductionType, StandardDeviationReduce},
    },
    parser::{parse_tokens, parse_expr, Token},
    shaderpreprocessor::{ShaderDefVal, ShaderProcessor, ShaderSpecs},
    utils::{default_device, inspect_buffers, read_buffer, FullComputePass},
};
use macros::*;
use pollster::FutureExt;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferDescriptor, BufferUsages,
};

#[test]
fn macro_inplace() {
    let mut pp = ShaderProcessor::default();
    add_directory!(pp, "tests/test_shaders");

    assert!(pp.all_shaders.contains_key(&"first".into()));
    assert!(pp.all_shaders.contains_key(&"second".into()));
}

#[test]
fn macro_return() {
    let pp = add_directory!("tests/test_shaders");
    assert!(pp.all_shaders.contains_key(&"first".into()));
    assert!(pp.all_shaders.contains_key(&"second".into()));
}

#[test]
fn for_loop() {
    let pp = add_directory!("tests/test_shaders");
    let specs = ShaderSpecs::new((1, 1, 1)).extend_defs(&[ShaderDefVal::UInt("HI".to_string(), 5)]);
    let shader = pp.process_by_key("for_loop", specs).unwrap();

    let answer = "@group(0) @binding(0)
var<storage, read_write> buffer: array<f32>;

@compute @workgroup_size(1u, 1u, 1u)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>){
\t0, 0u
\t0, 1u
\t0, 2u
\t1, 0u
\t1, 1u
\t1, 2u
\t2, 0u
\t2, 1u
\t2, 2u
\t3, 0u
\t3, 1u
\t3, 2u
\t4, 0u
\t4, 1u
\t4, 2u
}
";
    print!("{}", shader.source);
    assert_eq!(answer, shader.source);
}

#[test]
fn standard_deviation() {
    let (device, queue) = default_device().block_on().unwrap();
    let output = device.create_buffer(&BufferDescriptor {
        label: None,
        size: (std::mem::size_of::<f32>()) as _,
        usage: BufferUsages::STORAGE
            | BufferUsages::MAP_READ
            | BufferUsages::UNIFORM
            | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let specs = ShaderSpecs::new((256, 1, 1));

    let full = |n| {
        let contents: Vec<f32> = (0..n).map(|x| x as f32).collect();

        let inp = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&contents),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let temp = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (n * std::mem::size_of::<f32>()) as _,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let std = StandardDeviationReduce::new(
            &device,
            &inp,
            &temp,
            None,
            &output,
            n,
            2,
            specs.clone(),
            8,
        )
        .unwrap();

        let mut encoder = device.create_command_encoder(&Default::default());

        std.execute(&mut encoder);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::MaintainBase::Wait);
        let result: f32 = read_buffer(&device, &output)[0];
        let mean = contents.iter().sum::<f32>() / n as f32;
        let std = contents.iter().map(|ele| (ele - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
        let std = std.sqrt();

        println!(
            "n: {n}, rel_error: {}, true_std: {std}",
            (result - std).abs() / std
        );

        assert!((result - std).abs() / std < 0.0001);
    };

    for i in 980..1020 {
        full(i)
    }
    full(8);
    full(391939);
}

#[test]
fn preprocess() {
    let (device, queue) = default_device().block_on().unwrap();

    let pp = add_directory!("tests/test_shaders");

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (512 * 512 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let n = 1000;
    let tests = |it: usize| -> Option<Duration> {
        let wg = match it {
            0 => (16, 16, 1),
            1 => (256, 1, 1),
            2 => (1, 1, 1),
            _ => return None,
        };
        let specs = ShaderSpecs::new(wg)
            .direct_dispatcher(&[512, 512, 1])
            .extend_defs(&[
                ShaderDefVal::UInt("TEST".into(), 32),
                ShaderDefVal::UInt("N_COL".into(), 512),
            ]);

        let result = pp.process_by_key("first", specs).unwrap();

        let nonbound = result.build(&device);

        let bindings: [(u32, &wgpu::Buffer); 1] = [(0, &buffer)];

        let pass = FullComputePass::new(&device, nonbound, &bindings);

        let now = Instant::now();
        for _ in 0..n {
            let mut encoder = device.create_command_encoder(&Default::default());

            pass.execute(&mut encoder, &[]);

            queue.submit(Some(encoder.finish()));

            device.poll(wgpu::Maintain::Wait);
        }

        Some(now.elapsed())
    };

    let mut i = 0;
    while let Some(duration) = tests(i) {
        dbg!(duration);
        i += 1;
    }
}

#[test]
fn max() {
    let (device, queue) = default_device().block_on().unwrap();
    let output = device.create_buffer(&BufferDescriptor {
        label: None,
        size: (std::mem::size_of::<f32>()) as _,
        usage: BufferUsages::STORAGE
            | BufferUsages::MAP_READ
            | BufferUsages::UNIFORM
            | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let specs = ShaderSpecs::new((256, 1, 1));

    let full = |n: i32| {
        let contents: Vec<f32> = (-n..n).map(|x| x as f32).collect();

        let inp = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&contents),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let temp = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (3 * n as usize * std::mem::size_of::<f32>()) as _,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let ty = ReductionType::Min;

        let cpu_result = match ty {
            ReductionType::Min => contents.iter().min_by(|a, b| a.total_cmp(b)).unwrap(),
            ReductionType::Max => contents.iter().max_by(|a, b| a.total_cmp(b)).unwrap(),
            _ => panic!("test should be min or max"),
        };

        let reduction = Reduce::new(
            &device,
            &inp,
            Some(&temp),
            &output,
            (2 * n) as _,
            ty,
            false,
            8,
            specs.clone(),
            "".to_string(),
            "".to_string(),
            "".to_string(),
            None,
            24,
            "",
        )
        .unwrap();

        let mut encoder = device.create_command_encoder(&Default::default());

        reduction.execute(&mut encoder, &[]);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::MaintainBase::Wait);
        let result: f32 = read_buffer(&device, &output)[0];

        println!(
            "n: {n}, rel_error: {}, return: {result}, cpu_result: {cpu_result}",
            (result - cpu_result).abs()
        );

        assert!((result - cpu_result).abs() < 0.0001);
    };

    for i in 980..1020 {
        full(i)
    }
    full(8);
    full(391939);
}

#[test]
fn gaussian_smoothing() {
    let (device, queue) = default_device().block_on().unwrap();

    let full = |n: i32| {
        let contents: Vec<f32> = (-n..n).map(|x| x as f32).collect();

        let inp = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&contents),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let out = device.create_buffer(&BufferDescriptor {
            label: None,
            size: inp.size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let smoothing = GaussianSmoothing::new::<3>(&device, &[512, 512, 5000], &inp, &out, 5.0);
    };

    full(100);
}

#[test]
fn lexing() {
    // let data = include_str!("../src/operations/shaders/last_reduction.wgsl");
    let data = "1 #if a == 0 + 1 * 5 / 7 {  #if adsasd{3}}";
    // let data = "#if -a{}";
    let mut out = Vec::new();
    let now = std::time::Instant::now();
    for _ in 0..1e3 as usize {
        let (_input, output) = parse_tokens(data).unwrap();
        out = output;
    }
    dbg!(now.elapsed());
    dbg!(&out);

    let ser = bincode::serialize(&out).unwrap();
    dbg!(data.len());
    dbg!(ser.len());
    // dbg!(&ser);
    let de: Vec<Token> = bincode::deserialize(&ser).unwrap();
    // drop(ser);
    // dbg!(de);
    // dbg!(bincode::serialize(&out));
}
