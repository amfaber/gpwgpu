// #![cfg_attr(feature = "nightly", feature(portable_simd))]
#![allow(unused_imports)]
use std::{
    collections::HashMap,
    io::Read,
    time::{Duration, Instant},
};

use bytemuck::Pod;
use gpwgpu::{
    automatic_buffers::MemoryReq,
    operations::{
        convolutions::{GaussianLaplace, GaussianSmoothing},
        reductions::{Reduce, ReductionType, StandardDeviationReduce},
        simple::new_simple,
    },
    parser::{parse_tokens, process, trim_trailing_spaces, Definition, NestedFor, Token, parse_token_expr},
    shaderpreprocessor::{ShaderProcessor, ShaderSpecs},
    utils::{
        default_device, read_buffer, AccTime, DebugBundle, Dispatcher, Encoder, FullComputePass,
        InspectBuffer,
    },
};
use gpwgpu_macros::*;
use ndarray::{Array, Array3, Axis, Dim};
use pollster::FutureExt;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferDescriptor, BufferUsages,
};

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
            // n,
            2,
            specs.clone(),
            8,
        )
        .unwrap();

        let mut encoder = Encoder::new(&device);
        std.execute(&mut encoder, n as _);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::MaintainBase::Wait);
        let result: f32 = read_buffer(&device, &output, 0, None)[0];
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
            // (2 * n) as _,
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

        // let mut encoder = device.create_command_encoder(&Default::default());
        let mut encoder = Encoder::new(&device);

        reduction.execute(&mut encoder, 2 * n as u32, &[]);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::MaintainBase::Wait);
        let result: f32 = read_buffer(&device, &output, 0, None)[0];

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
#[should_panic]
fn gaussian_smoothing_2d() {
    let (device, queue) = default_device().block_on().unwrap();

    let (data, shape) = load_lion();

    let inp = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let temp = device.create_buffer(&BufferDescriptor {
        label: None,
        size: inp.size(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let out = device.create_buffer(&BufferDescriptor {
        label: None,
        size: inp.size(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // let readable = device.create_buffer(&BufferDescriptor {
    //     label: None,
    //     size: inp.size(),
    //     usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    //     mapped_at_creation: false,
    // });

    let smoothing = GaussianSmoothing::new(&device, shape, &inp, &temp, &out).unwrap();
    // let mut encoder = device.create_command_encoder(&Default::default());
    let mut encoder = Encoder::new(&device);
    encoder.set_debug_bundle(DebugBundle {
        device: &device,
        queue: &queue,
        inspects: vec![
            InspectBuffer::new(&inp, None, "Input"),
            InspectBuffer::new(&temp, None, "Temp"),
            InspectBuffer::new(&out, None, "Out"),
        ],
        save_path: "tests/dumps".into(),
        create_py: false,
    });
    smoothing.execute(&mut encoder, shape, [5.0; 2]);
    encoder.inspect_buffers().unwrap();
}

fn _load_cells() -> (Array3<f32>, [i32; 3]) {
    let mut file = std::fs::File::open("tests/images/cells3d.bin").unwrap();
    let mut data_init = Vec::<u16>::with_capacity(
        file.metadata().unwrap().len() as usize / std::mem::size_of::<u16>(),
    );
    unsafe {
        data_init.set_len(data_init.capacity());
    }

    file.read(bytemuck::cast_slice_mut(&mut data_init[..]))
        .unwrap();
    let data = Array::from_shape_vec((60, 2, 256, 256), data_init).unwrap();
    let channel = 1;
    let data_owned = data.index_axis(Axis(1), channel).mapv(|val| val as f32);
    let shape: [_; 3] = data_owned
        .shape()
        .iter()
        .map(|&x| x as i32)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    (data_owned, shape)
}

struct DefaultBuffers {
    inp: wgpu::Buffer,
    temp: wgpu::Buffer,
    temp2: wgpu::Buffer,
    out: wgpu::Buffer,
    readable: wgpu::Buffer,
}

impl DefaultBuffers {
    fn new<T: Pod>(device: &wgpu::Device, data: &[T]) -> Self {
        let inp = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let temp = device.create_buffer(&BufferDescriptor {
            label: None,
            size: inp.size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let temp2 = device.create_buffer(&BufferDescriptor {
            label: None,
            size: inp.size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out = device.create_buffer(&BufferDescriptor {
            label: None,
            size: inp.size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readable = device.create_buffer(&BufferDescriptor {
            label: None,
            size: inp.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            inp,
            temp,
            temp2,
            out,
            readable,
        }
    }
}

fn create_blob(shape: [i32; 3], r: f32) -> Array3<f32> {
    let [im, jm, km] = shape.map(|x| x as f32);

    let data = Array::from_shape_fn(shape.map(|x| x as usize), |(i, j, k)| {
        let (x, y, z) = (i as f32 - im / 2., j as f32 - jm / 2., k as f32 - km / 2.);
        let this_r = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
        if this_r < r {
            1.0
        } else {
            0.0
        }
    });
    data
}

#[test]
fn smoothing_3d() {
    let (device, queue) = default_device().block_on().unwrap();

    // let (data_owned, shape) = load_cells();

    let shape = [100, 100, 100];
    let data_owned = create_blob(shape, 3.);

    let data = data_owned.as_slice().unwrap();

    let bufs = DefaultBuffers::new(&device, data);

    // let mut encoder = device.create_command_encoder(&Default::default());
    let mut encoder = Encoder::new(&device);
    if false {
        let smoothing =
            GaussianSmoothing::new(&device, shape, &bufs.inp, &bufs.temp, &bufs.out).unwrap();
        smoothing.execute(&mut encoder, shape, [5.0, 5.0, 2.0]);
    } else {
        let laplace = GaussianLaplace::new(
            &device,
            shape,
            &bufs.inp,
            &bufs.temp,
            &bufs.temp2,
            &bufs.out,
        )
        .unwrap();
        laplace.execute(&mut encoder, shape, [4.0, 4.0, 4.0]);
    }

    encoder.copy_buffer_to_buffer(&bufs.out, 0, &bufs.readable, 0, bufs.inp.size());

    queue.submit(Some(encoder.finish()));

    let result = read_buffer::<u8>(&device, &bufs.readable, 0, None);
    std::fs::write("tests/dumps/3d_smoothed.bin", &result).unwrap();

    // println!("{}", shape);

    let shape0 = shape[0];
    let shape1 = shape[1];
    let shape2 = shape[2];

    std::process::Command::new("python")
        .arg("-c")
        .arg(format!(r#"\
import numpy as np
import tifffile
arr = np.fromfile("tests/dumps/3d_smoothed.bin", dtype = "float32").reshape({shape0}, {shape1}, {shape2})
arr = arr.astype("uint16")
tifffile.imsave("tests/dumps/3d_smoothed.tif", arr)"#
        )).status().unwrap();
}

fn load_lion() -> (Vec<f32>, [i32; 2]) {
    let file = std::fs::File::open("tests/images/grey_lion.tiff").unwrap();

    let mut decoder = tiff::decoder::Decoder::new(file).unwrap();

    let (width, height) = decoder.dimensions().unwrap();

    let shape = [height as i32, width as i32];

    let Ok(tiff::decoder::DecodingResult::U8(data)) = decoder.read_image()
        else { panic!("couldn't read image") };

    let data = data.iter().map(|&x| x as f32).collect::<Vec<_>>();
    (data, shape)
}

#[test]
#[should_panic]
fn laplace() {
    let (device, queue) = default_device().block_on().unwrap();

    let (data, shape) = load_lion();

    let inp = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let temp = device.create_buffer(&BufferDescriptor {
        label: None,
        size: inp.size(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let temp2 = device.create_buffer(&BufferDescriptor {
        label: None,
        size: inp.size(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let out = device.create_buffer(&BufferDescriptor {
        label: None,
        size: inp.size(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let laplace = GaussianLaplace::new(&device, shape, &inp, &temp, &temp2, &out).unwrap();
    // let mut encoder = device.create_command_encoder(&Default::default());
    let mut encoder = Encoder::new(&device);
    encoder.set_debug_bundle(DebugBundle {
        device: &device,
        queue: &queue,
        inspects: vec![
            InspectBuffer::new(&inp, None, "Input"),
            InspectBuffer::new(&temp, None, "Temp"),
            InspectBuffer::new(&out, None, "Out"),
        ],
        save_path: "tests/dumps".into(),
        create_py: false,
    });
    laplace.execute(&mut encoder, shape, [5.0; 2]);
    encoder.inspect_buffers().unwrap();
}

#[test]
fn parser_test() {
    // let data = include_str!("../src/operations/shaders/last_reduction.wgsl");
    // let data = "1 #TEST #if a == 0 + 1 * 5 / 7 {  #if adsasd{3}}";
    let data = "\
// #for i in -RINT..=RINT{
#if idk == 3{
    This gets printed if idk is 3
}
#else {
    this is the else block
}
// }";
    // let data = "";

    let mut out = Vec::new();
    let reps = 1e4 as usize;
    let now = std::time::Instant::now();
    for _ in 0..reps {
        let (_input, output) = parse_tokens(data).unwrap();
        out = output;
    }
    dbg!(now.elapsed());
    dbg!(&out);

    let ser = bincode::serialize(&out).unwrap();
    dbg!(data.len());
    dbg!(ser.len());
    let de_time = std::time::Instant::now();
    for _ in 0..reps {
        bincode::deserialize::<Vec<Token>>(&ser).unwrap();
    }
    dbg!(de_time.elapsed());

    println!(
        "{}",
        process(
            out.clone(),
            |s| {
                if s == "RINT" {
                    Some(Definition::Int(3))
                } else if s == "idk" {
                    Some(Definition::Int(3))
                } else {
                    None
                }
            },
            |_| None
        )
        .unwrap()
    );

    println!("--------------------");

    println!(
        "{}",
        process(
            out,
            |s| {
                if s == "RINT" {
                    Some(Definition::Int(3))
                } else if s == "idk" {
                    Some(Definition::Int(4))
                } else {
                    None
                }
            },
            |_| None
        )
        .unwrap()
    );
    // drop(ser);
    // dbg!(de);
    // dbg!(bincode::serialize(&out));
}

#[test]
fn nested_for() {
    let data = "#nest I = N { for (var i#I: i32 = #I; i#I < 2*#I; i += 1) }
    #pre { 1+1; }
    #inner { #concat I in 0..N {pow(f32(i#I), 2.)} { + } }";

    let (_input, tokens) = parse_tokens(data).unwrap();
    dbg!(&tokens);

    println!(
        "{}",
        process(tokens, |_s| Some(Definition::Int(3)), |_| None).unwrap()
    );
}

#[test]
fn not() {
    let data = "#if !TEST{
        Hi
    }
    outside";
    let (_input, tokens) = parse_tokens(data).unwrap();

    dbg!(
        process(tokens.clone(), |_s| Some(Definition::Bool(true)), |_| None)
            .unwrap()
            .as_str()
    );

    dbg!(process(tokens, |_s| Some(Definition::Bool(false)), |_| None).unwrap());
}

#[test]
fn parse_expr() {
    let data = "#expr{ 1 + N }";

    let (_input, tokens) = parse_tokens(data).unwrap();

    dbg!(
        process(tokens.clone(), |_s| Some(Definition::Int(5)), |_| None)
            .unwrap()
            .as_str()
    );
}

#[test]
fn import_testing() {
    let data1 = "#export test {yoyoyo #N}".to_string();
    let data2 = "#import test".to_string();

    let hashmap = HashMap::from([("1".into(), data1), ("2".into(), data2)]);
    let processor = ShaderProcessor::from_shader_hashmap(&hashmap).unwrap();
    dbg!(process(
        processor.shaders.get("2").unwrap().0.clone(),
        |_| Some(Definition::Int(2)),
        |import| processor.exports.get(&import).cloned()
    )
    .unwrap());
}

#[test]
fn encoder_vs_pass() {
    let (device, queue) = default_device().block_on().unwrap();

    let buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 512 * 512 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readable = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 512 * 512 * 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let full = new_simple(
        &device,
        512 * 512,
        gpwgpu::operations::simple::OperationType::Add,
        &buffer,
        gpwgpu::operations::simple::UnaryBinary::Unary(1.),
        None,
    )
    .unwrap();

    let n = 100000;

    let mut time = AccTime::new(true);
    let mut encoder = Encoder::new(&device);
    // time.start();
    for _ in 0..n {
        full.execute(&mut encoder, &[]);
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &readable, 0, readable.size());
    queue.submit(Some(encoder.finish()));
    time.start();
    device.poll(wgpu::Maintain::Wait);
    time.stop();
    dbg!(time);

    let data = read_buffer::<f32>(&device, &readable, 0, None);
    dbg!(&data[..10]);

    let mut time = AccTime::new(true);
    let mut encoder = Encoder::new(&device);
    let mut cpass = encoder.begin_compute_pass(&Default::default());
    time.start();
    for _ in 0..n {
        cpass.set_bind_group(0, &full.bindgroup, &[]);
        cpass.set_pipeline(&full.pipeline.compute_pipeline);
        let Dispatcher::Direct([x, y, z]) = full.pipeline.dispatcher.as_ref().unwrap() else { panic!() };
        cpass.dispatch_workgroups(*x, *y, *z);
    }
    drop(cpass);
    encoder.copy_buffer_to_buffer(&buffer, 0, &readable, 0, readable.size());
    queue.submit(Some(encoder.finish()));
    time.start();
    device.poll(wgpu::Maintain::Wait);
    time.stop();
    dbg!(time);
    
    let data = read_buffer::<f32>(&device, &readable, 0, None);
    dbg!(&data[..10]);
}


#[test]
fn dim_test(){
    let idk = Dim([0; 2]) - Dim([1; 2]);
    dbg!(idk);
}

// use std::simd::Simd;
// #[test]
// fn simd_copy(){
//     let total = 100_000_000;
//     let idk = vec![5.; total];
//     // let mut target = Vec::with_capacity(total);

//     let now = std::time::Instant::now();
//     // target.extend_from_slice(&idk);
//     let target = idk.into_iter().sum::<f32>();
//     dbg!(now.elapsed());
//     std::hint::black_box(target);

//     const N: usize = 2;

//     let idk_simd = vec![Simd::from([5f32; N]); total/N];
//     // let mut target = Vec::with_capacity(total / N);
    
//     let now = std::time::Instant::now();
//     // target.extend_from_slice(&idk_simd);
//     let target = idk_simd.into_iter().sum::<Simd<f32, N>>();
//     dbg!(now.elapsed());
//     std::hint::black_box(target);
// }

// #[test]
// fn get_available_mem(){
//     let instance = wgpu::Instance::default();
//     let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions{
//         power_preference: wgpu::PowerPreference::HighPerformance,
//         ..Default::default()
//     }).block_on().unwrap();

//     dbg!(adapter.limits());
//     // for adapter in instance.enumerate_adapters(wgpu::Backends::all()){
//     //     // drop(adapter)
//     //     // dbg!(adapter.get_info());
//     // }
// }

#[test]
fn primes_time_test(){
    use slow_primes::Primes;

    let gb_size = 2.50031231;
    let n_wg = gb_size * (1 << 30) as f64 / 4. / 256.;
    
    let now = std::time::Instant::now();
    let primes = Primes::sieve((n_wg.sqrt() * 1.1) as usize);

    
    let mut running = [1, 1, 1];
    let mut keep_going = true;
    let mut offset = 0;
    while keep_going{
        let factors = primes.factor(n_wg as usize + offset).unwrap();
        dbg!(&factors);
        let mut idx = 0;
        let mut iter = factors.into_iter();
        keep_going = loop{
            let Some((base, exp)) = iter.next() else { break false };
            if base > (1<<16){
                offset += 1;
                break true
            }
            for _ in 0..exp{
                if running[idx] * base < (1<<16){
                    running[idx] *= base;
                } else {
                    idx += 1;
                    running[idx] *= base;
                }
            }
        }
    }
    dbg!(now.elapsed());
    dbg!(running);
    dbg!(offset);
    // dbg!(factors);
}

#[test]
fn expr_test(){
    let data = "#expr{YO}";

    dbg!(parse_tokens(data).unwrap());
}
