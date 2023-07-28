#![allow(unused_imports)]
use std::{time::{Duration, Instant}, collections::HashMap, io::Read};

use bytemuck::Pod;
use gpwgpu::{
    operations::{
        convolutions::{GaussianSmoothing, GaussianLaplace},
        reductions::{Reduce, ReductionType, StandardDeviationReduce},
    },
    parser::{parse_tokens, Token, process, Definition, trim_trailing_spaces, NestedFor},
    shaderpreprocessor::{ShaderProcessor, ShaderSpecs},
    utils::{default_device, inspect_buffers, read_buffer, FullComputePass}, automatic_buffers::MemoryReq,
};
use macros::*;
use ndarray::{Array, Axis, Array3};
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

    let readable = device.create_buffer(&BufferDescriptor {
        label: None,
        size: inp.size(),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    

    let smoothing = GaussianSmoothing::new(&device, shape, &inp, &temp, &out).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    smoothing.execute(&mut encoder, [5.0; 2]);
    inspect_buffers(
        &[
            &inp,
            &temp,
            &out,
        ],
        &readable,
        &queue,
        &mut encoder,
        &device,
        "tests/dumps",
    );

}

fn _load_cells() -> (Array3<f32>, [i32; 3]){
    let mut file = std::fs::File::open("tests/images/cells3d.bin").unwrap();
    let mut data_init = Vec::<u16>::with_capacity(
        file.metadata().unwrap().len() as usize / std::mem::size_of::<u16>()
    );
    unsafe{
        data_init.set_len(data_init.capacity());
    }
    
    file.read(bytemuck::cast_slice_mut(&mut data_init[..])).unwrap();
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

struct DefaultBuffers{
    inp: wgpu::Buffer,
    temp: wgpu::Buffer,
    temp2: wgpu::Buffer,
    out: wgpu::Buffer,
    readable: wgpu::Buffer,
}

impl DefaultBuffers{
    fn new<T: Pod>(device: &wgpu::Device, data: &[T]) -> Self{
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
    
        Self{
            inp,
            temp,
            temp2,
            out,
            readable,
        }
        
    }
}

fn create_blob(shape: [i32; 3], r: f32) -> Array3<f32>{
    let [im, jm, km] = shape.map(|x| x as f32);
    
    let data = Array::from_shape_fn(shape.map(|x| x as usize), |(i, j, k)|{
        let (x, y, z) = (i as f32 - im/2., j as f32 - jm/2., k as f32 - km/2.);
        let this_r = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
        if this_r < r{
            1.0
        } else {
            0.0
        }
    });
    data
}

#[test]
fn smoothing_3d(){

    let (device, queue) = default_device().block_on().unwrap();
    
    // let (data_owned, shape) = load_cells();

    let shape = [100, 100, 100];
    let data_owned = create_blob(shape, 3.);
    
    let data = data_owned.as_slice().unwrap();

    let bufs = DefaultBuffers::new(&device, data);
    
    let mut encoder = device.create_command_encoder(&Default::default());
    if false{
        let smoothing = GaussianSmoothing::new(&device, shape, &bufs.inp, &bufs.temp, &bufs.out).unwrap();
        smoothing.execute(&mut encoder, [5.0, 5.0, 2.0]);
    } else {
        let laplace = GaussianLaplace::new(&device, shape, &bufs.inp, &bufs.temp, &bufs.temp2, &bufs.out).unwrap();
        laplace.execute(&mut encoder, [4.0, 4.0, 4.0]);
    }

    
    encoder.copy_buffer_to_buffer(
        &bufs.out,
        0,
        &bufs.readable,
        0,
        bufs.inp.size()
    );

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

fn load_lion() -> (Vec<f32>, [i32; 2]){
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

    let readable = device.create_buffer(&BufferDescriptor {
        label: None,
        size: inp.size(),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    

    let laplace = GaussianLaplace::new(&device, shape, &inp, &temp, &temp2, &out).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    laplace.execute(&mut encoder, [5.0; 2]);
    inspect_buffers(
        &[
            &inp,
            &temp,
            &out,
        ],
        &readable,
        &queue,
        &mut encoder,
        &device,
        "tests/dumps",
    );

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
    for _ in 0..reps{
        let (_input, output) = parse_tokens(data).unwrap();
        out = output;
    }
    dbg!(now.elapsed());
    dbg!(&out);

    let ser = bincode::serialize(&out).unwrap();
    dbg!(data.len());
    dbg!(ser.len());
    let de_time = std::time::Instant::now();
    for _ in 0..reps{
        bincode::deserialize::<Vec<Token>>(&ser).unwrap();
    }
    dbg!(de_time.elapsed());

    println!("{}", process(out.clone(), |s|{
        if s == "RINT"{
            Some(Definition::Int(3))
        } else if s == "idk"{
            Some(Definition::Int(3))
        } else {
            None
        }
    }).unwrap());

    println!("--------------------");

    println!("{}", process(out, |s|{
        if s == "RINT"{
            Some(Definition::Int(3))
        } else if s == "idk"{
            Some(Definition::Int(4))
        } else {
            None
        }
    }).unwrap());
    // drop(ser);
    // dbg!(de);
    // dbg!(bincode::serialize(&out));
}

#[test]
fn nested_for(){
    let data = "#nest I = N { for (var i#I: i32 = #I; i#I < 2*#I; i += 1) }
    #pre { 1+1; }
    #inner { #concat I in 0..N {pow(f32(i#I), 2.)} { + } }";

    
    let (_input, tokens) = parse_tokens(data).unwrap();
    dbg!(&tokens);

    println!("{}", process(tokens, |_s| Some(Definition::Int(3))).unwrap());
}

#[test]
fn not(){
    let data = "#if !TEST{
        Hi
    }
    outside";
    let (_input, tokens) = parse_tokens(data).unwrap();

    dbg!(process(tokens.clone(), |_s| Some(Definition::Bool(true))).unwrap().as_str());

    dbg!(process(tokens, |_s| Some(Definition::Bool(false))).unwrap());
}

#[test]
fn parse_expr(){
    let data = "#expr{ 1 + N }";

    let (_input, tokens) = parse_tokens(data).unwrap();

    dbg!(process(tokens.clone(), |_s| Some(Definition::Int(5))).unwrap().as_str());

}
