use once_cell::sync::Lazy;
// use crate::gpu_setup::GpuState;
use regex::{self, Regex};
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Write,
    mem::size_of,
    ops::{Bound, Deref, DerefMut},
    path::{PathBuf, Path},
    rc::Rc,
};
use wgpu::{self, util::DeviceExt, CommandEncoder, MAP_ALIGNMENT};

use crate::shaderpreprocessor::NonBoundPipeline;

struct BufferPadding {
    unpadded_bytes_per_row: u32,
    padded_bytes_per_row: u32,
}

impl BufferPadding {
    fn new(width: u32) -> Self {
        let bytes_per_pixel = std::mem::size_of::<u32>() as u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let padded_bytes_per_row =
            wgpu::util::align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        Self {
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}

pub fn inspect_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    encoder: &mut wgpu::CommandEncoder,
    path: impl AsRef<Path>,
) -> ! {
    let mut encoder = std::mem::replace(encoder, device.create_command_encoder(&Default::default()));

    let padding = BufferPadding::new(texture.width());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padding.padded_bytes_per_row * texture.height() * texture.depth_or_array_layers()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // let rows_per_image = match (texture.dimension(), texture.depth_or_array_layers()){
    //     (wgpu::TextureDimension::D2, ) => ,
    //     _ => None
    // };

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::ImageCopyBufferBase {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padding.padded_bytes_per_row),
                rows_per_image: Some(texture.height()),
            },
        },
        texture.size(),
    );

    queue.submit(Some(encoder.finish()));
    
    let slc = buffer.slice(..);

    slc.map_async(wgpu::MapMode::Read, |_| {});

    device.poll(wgpu::Maintain::Wait);

    let mut data = Vec::with_capacity((texture.width() * texture.height() * texture.depth_or_array_layers() * 4) as _);
    for padded in slc.get_mapped_range().chunks(padding.unpadded_bytes_per_row as usize){
        data.extend_from_slice(&padded[..padding.unpadded_bytes_per_row as usize]);
    }

    let mut size_name = path.as_ref().to_path_buf();
    let mut name = path.as_ref().file_name().unwrap().to_string_lossy().to_string();
    name.push_str("_size");
    size_name.set_file_name(name);
    size_name.set_extension("txt");
    
    std::fs::write(path, data).unwrap();
    std::fs::write(size_name, format!("{}, {}, {}", texture.depth_or_array_layers(), texture.height(), texture.width())).unwrap();
    
    panic!("Inspected texture")
}

/// Convenience function to read a Pod type from a mappable buffer.
pub fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    mappable_buffer: &wgpu::Buffer,
    offset: u64,
    n_items: Option<u64>,
) -> Vec<T> {
    let byte_size = size_of::<T>() as wgpu::BufferAddress;
    let end = match n_items {
        Some(n) => Bound::Excluded(offset + n * byte_size),
        None => Bound::Unbounded,
    };
    let slice = mappable_buffer.slice((Bound::Included(offset), end));
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    device.poll(wgpu::MaintainBase::Wait);
    receiver.recv().unwrap().unwrap();
    let view = slice.get_mapped_range();

    let out = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    mappable_buffer.unmap();
    out
}

/// Same as [read_buffer], but does a copy from the data buffer to the mappable buffer before
/// mapping and reading from the mappable buffer.
pub fn read_from_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data_buffer: &wgpu::Buffer,
    mappable_buffer: &wgpu::Buffer,
) -> Vec<T> {
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(data_buffer, 0, mappable_buffer, 0, data_buffer.size());
    queue.submit(Some(encoder.finish()));
    read_buffer::<T>(device, mappable_buffer, 0, None)
}

/// A convenience function to read a number of items a buffer with that number being
/// by the contents of another buffer.
///
/// Has not been performance tested, but is useful for debugging.
pub fn read_n_from_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    counter_buffer: &wgpu::Buffer,
    data_buffer: &wgpu::Buffer,
    mappable_buffer: &wgpu::Buffer,
) -> Vec<T> {
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(counter_buffer, 0, mappable_buffer, 0, 4);
    encoder.copy_buffer_to_buffer(
        data_buffer,
        0,
        mappable_buffer,
        MAP_ALIGNMENT,
        data_buffer.size(),
    );
    queue.submit(Some(encoder.finish()));

    let n = read_buffer::<u32>(device, mappable_buffer, 0, Some(1))[0];
    read_buffer::<T>(device, mappable_buffer, MAP_ALIGNMENT, Some(n as u64))
}

pub trait BindingGroup {
    fn binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: Option<&str>,
    ) -> wgpu::BindGroup;
}

impl<'a> BindingGroup for [(u32, &'a wgpu::Buffer)] {
    fn binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: Option<&str>,
    ) -> wgpu::BindGroup {
        let entries = self
            .iter()
            .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: entries.as_slice(),
        })
    }
}

impl<'a, const N: usize> BindingGroup for [(u32, &'a wgpu::Buffer); N] {
    fn binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: Option<&str>,
    ) -> wgpu::BindGroup {
        let entries = self
            .iter()
            .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: entries.as_slice(),
        })
    }
}

impl<'a> BindingGroup for HashMap<u32, &'a wgpu::Buffer> {
    fn binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: Option<&str>,
    ) -> wgpu::BindGroup {
        let entries = self
            .iter()
            .map(|(&binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: entries.as_slice(),
        })
    }
}

impl<'a> BindingGroup for [wgpu::BindGroupEntry<'a>] {
    fn binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: Option<&str>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: self,
        })
    }
}

impl<'a, const N: usize> BindingGroup for [wgpu::BindGroupEntry<'a>; N] {
    fn binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: Option<&str>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: self,
        })
    }
}

impl<'a> BindingGroup for Vec<(u32, &'a wgpu::Buffer)> {
    fn binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: Option<&str>,
    ) -> wgpu::BindGroup {
        self.as_slice().binding_group(device, layout, label)
    }
}

impl<'a> BindingGroup for Vec<wgpu::BindGroupEntry<'a>> {
    fn binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: Option<&str>,
    ) -> wgpu::BindGroup {
        self.as_slice().binding_group(device, layout, label)
    }
}

impl<'a, T: BindingGroup> BindingGroup for &T {
    fn binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: Option<&str>,
    ) -> wgpu::BindGroup {
        <T as BindingGroup>::binding_group(*self, device, layout, label)
    }
}

pub async fn default_device() -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
    let instance = wgpu::Instance::new(Default::default());
    let adapter = instance.request_adapter(&Default::default());
    let mut device_desc = wgpu::DeviceDescriptor::default();
    device_desc.limits.max_push_constant_size = 64;
    device_desc.features =
        wgpu::Features::MAPPABLE_PRIMARY_BUFFERS | wgpu::Features::PUSH_CONSTANTS;
    device_desc.limits.max_storage_buffers_per_shader_stage = 12;
    adapter
        .await
        .ok_or(wgpu::RequestDeviceError)?
        .request_device(&device_desc, None)
        .await
}

pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
}

#[derive(Debug, Clone)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub x_name: Cow<'static, str>,
    pub y_name: Cow<'static, str>,
    pub z_name: Cow<'static, str>,
}

impl WorkgroupSize {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            x,
            y,
            z,
            x_name: "WG_X".into(),
            y_name: "WG_Y".into(),
            z_name: "WG_Z".into(),
        }
    }
}

impl From<(u32, u32, u32)> for WorkgroupSize {
    fn from(value: (u32, u32, u32)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl From<[u32; 3]> for WorkgroupSize {
    fn from(value: [u32; 3]) -> Self {
        Self::new(value[0], value[1], value[2])
    }
}

#[derive(Debug)]
pub struct IndirectDispatcher {
    dispatcher: wgpu::Buffer,
    resetter: wgpu::Buffer,
}

#[derive(Debug, Clone)]
pub enum Dispatcher<'a> {
    Direct([u32; 3]),
    Indirect(Rc<IndirectDispatcher>),
    IndirectBorrowed {
        dispatcher: &'a wgpu::Buffer,
        resetter: &'a wgpu::Buffer,
    },
}

impl<'a> Dispatcher<'a> {
    pub fn new_direct(dims: &[u32; 3], wgsize: &WorkgroupSize) -> Self {
        let mut n_workgroups = [0, 0, 0];
        let wgsize = [wgsize.x, wgsize.y, wgsize.z];
        for i in 0..3 {
            n_workgroups[i] = (dims[i] + wgsize[i] - 1) / wgsize[i];
        }
        Self::Direct(n_workgroups)
    }

    pub fn new_indirect(device: &wgpu::Device, default: wgpu::util::DispatchIndirect) -> Self {
        let dispatcher = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: default.as_bytes(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let resetter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: default.as_bytes(),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        Self::Indirect(Rc::new(IndirectDispatcher {
            dispatcher,
            resetter,
        }))
    }

    pub fn reset_indirect(&self, encoder: &mut wgpu::CommandEncoder) {
        let (dispatcher, resetter) = match self {
            Self::Indirect(indirect) => (&indirect.dispatcher, &indirect.resetter),
            Self::IndirectBorrowed {
                dispatcher,
                resetter,
            } => (dispatcher as &wgpu::Buffer, resetter as &wgpu::Buffer),
            Self::Direct(_) => return,
        };
        encoder.copy_buffer_to_buffer(resetter, 0, dispatcher, 0, (size_of::<u32>() * 3) as _)
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Indirect(indirect) => &indirect.dispatcher,
            Self::IndirectBorrowed { dispatcher, .. } => dispatcher,
            Self::Direct(_) => panic!("Tried to get buffer of a direct dispatcher."),
        }
    }
}

#[derive(Clone)]
pub struct DebugBundle<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub inspects: Vec<InspectBuffer<'a>>,
    pub save_path: PathBuf,
    pub create_py: bool,
}

pub struct DebugEncoder<'a> {
    pub encoder: CommandEncoder,
    pub debug_bundle: Option<DebugBundle<'a>>,
    pub debug_active: (bool, bool),
}

#[must_use]
pub enum InspectStatus {
    Inactive,
    Success,
    NoDebugBundle,
    NoBuffers,
}

impl InspectStatus {
    #[track_caller]
    pub fn unwrap(&self) {
        match self {
            Self::Inactive => (),
            Self::Success => panic!("Deliberate panic after inspecting buffers"),
            Self::NoDebugBundle => panic!("Inspect buffers was called without a debug bundle"),
            Self::NoBuffers => panic!("The inspect buffers vector is empty"),
        }
    }
}

impl<'a> DebugEncoder<'a> {
    pub fn new(device: &wgpu::Device) -> Self {
        let encoder = device.create_command_encoder(&Default::default());
        Self {
            encoder,
            debug_bundle: None,
            debug_active: (false, false),
        }
    }
    pub fn set_debug_bundle(&mut self, debug_bundle: DebugBundle<'a>) {
        self.debug_bundle = Some(debug_bundle);
    }
    pub fn activate(&mut self) {
        self.debug_active.0 = true;
    }
    pub fn inactivate(&mut self) {
        self.debug_active.0 = false;
    }
    pub fn activate_primed(&mut self) {
        if self.debug_active.1 {
            self.debug_active.0 = true
        }
    }
    pub fn prime(&mut self) {
        self.debug_active.1 = true;
    }
    pub fn unprime(&mut self) {
        self.debug_active.1 = false;
    }
    pub fn finish(self) -> wgpu::CommandBuffer {
        self.encoder.finish()
    }
    pub fn submit(self, queue: &wgpu::Queue) -> wgpu::SubmissionIndex {
        queue.submit(Some(self.encoder.finish()))
    }

    pub fn inspect_buffers(&mut self) -> InspectStatus {
        let DebugEncoder {
            encoder,
            debug_bundle,
            debug_active,
        } = self;

        if !debug_active.0 {
            return InspectStatus::Inactive;
        }

        let Some(DebugBundle {
            device,
            queue,
            inspects,
            save_path,
            create_py,
        }) = debug_bundle else {
            return InspectStatus::NoDebugBundle
        };

        let encoder =
            std::mem::replace(encoder, device.create_command_encoder(&Default::default()));
        queue.submit(Some(encoder.finish()));

        let max_size = match inspects.iter().max_by_key(|&inspect| inspect.size) {
            Some(inspect_buffer) => inspect_buffer.size,
            None => return InspectStatus::NoBuffers,
        };

        let mappable_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mappable buffer for inspect"),
            size: max_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for InspectBuffer { buffer, size, name } in inspects.iter() {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.clear_buffer(&mappable_buffer, 0, None);
            encoder.copy_buffer_to_buffer(buffer, 0, &mappable_buffer, 0, *size);
            queue.submit(Some(encoder.finish()));
            let slice = mappable_buffer.slice(..*size);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range().to_vec();
            std::fs::write(&save_path.join(format!("{name}.bin")), &data).unwrap();
            mappable_buffer.unmap();
        }

        if *create_py {
            let contents = create_py_file(inspects.iter());
            std::fs::write(&save_path.join("load.py"), &contents).unwrap();
        }

        InspectStatus::Success
    }
}

impl<'a> Deref for DebugEncoder<'a> {
    type Target = wgpu::CommandEncoder;
    fn deref(&self) -> &Self::Target {
        &self.encoder
    }
}

impl<'a> DerefMut for DebugEncoder<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.encoder
    }
}

pub struct NormalEncoder(CommandEncoder);

impl NormalEncoder {
    pub fn new(device: &wgpu::Device) -> Self {
        let encoder = device.create_command_encoder(&Default::default());
        Self(encoder)
    }
    pub fn finish(self) -> wgpu::CommandBuffer {
        self.0.finish()
    }
    pub fn submit(self, queue: &wgpu::Queue) -> wgpu::SubmissionIndex {
        queue.submit(Some(self.0.finish()))
    }
}

impl Deref for NormalEncoder {
    type Target = wgpu::CommandEncoder;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for NormalEncoder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub type Encoder<'a> = DebugEncoder<'a>;
// pub type Encoder = NormalEncoder;

#[derive(Debug)]
pub struct FullComputePass {
    pub bindgroup: wgpu::BindGroup,
    pub pipeline: Rc<NonBoundPipeline>,
}

impl FullComputePass {
    pub fn new<'a>(
        device: &wgpu::Device,
        pipeline: Rc<NonBoundPipeline>,
        bindgroup: &impl BindingGroup,
    ) -> Self {
        let bindgroup = bindgroup.binding_group(
            device,
            &pipeline.bind_group_layout,
            pipeline.label.as_deref(),
        );

        Self {
            bindgroup,
            pipeline,
        }
    }

    pub fn execute_with_dispatcher(
        &self,
        encoder: &mut Encoder,
        push_constants: &[u8],
        dispatcher: &Dispatcher,
    ) {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &self.bindgroup, &[]);
        cpass.set_pipeline(&self.pipeline.compute_pipeline);
        if push_constants.len() > 0 {
            cpass.set_push_constants(0, push_constants);
        }
        match dispatcher {
            Dispatcher::Direct(ref wg_n) => {
                cpass.dispatch_workgroups(wg_n[0], wg_n[1], wg_n[2]);
            }
            Dispatcher::Indirect(indirect) => {
                cpass.dispatch_workgroups_indirect(&indirect.dispatcher, 0);
            }
            Dispatcher::IndirectBorrowed { dispatcher, .. } => {
                cpass.dispatch_workgroups_indirect(dispatcher, 0);
            }
        }
    }

    pub fn execute(&self, encoder: &mut Encoder, push_constants: &[u8]) {
        self.execute_with_dispatcher(
            encoder,
            push_constants,
            self.pipeline.dispatcher.as_ref().expect(r#"FullComputePass must have a dispatcher to use "execute". Either add a dispatcher, or use "execute_with_dispatcher" instead."#),
        );
    }

    pub fn reset_indirect(&self, encoder: &mut wgpu::CommandEncoder) {
        match self.pipeline.dispatcher {
            Some(ref dispatcher) => {
                dispatcher.reset_indirect(encoder);
            }
            None => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct InspectBuffer<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub size: wgpu::BufferAddress,
    pub name: String,
}

impl<'a> InspectBuffer<'a> {
    pub fn new(
        buf: &'a wgpu::Buffer,
        size: Option<wgpu::BufferAddress>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            buffer: buf,
            size: size.unwrap_or(buf.size()),
            name: name.into(),
        }
    }
}

fn create_py_file<'a>(bufs: impl IntoIterator<Item = &'a InspectBuffer<'a>>) -> String {
    let mut file = "import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
here = Path(__file__).parent
__all__ = [\"np\", \"plt\"]

"
    .to_string();
    for InspectBuffer {
        buffer: _,
        size: _,
        name,
    } in bufs
    {
        write!(
            &mut file,
            "{name} = np.fromfile(here/\"{name}.bin\", dtype = \"float32\")
__all__.append(\"{name}\")
"
        )
        .unwrap();
    }
    file
}

pub static SHADER_BINDGROUP_INFER: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"@binding\((?P<idx>\d+)\)\s*var<(?P<type>.*?)>").unwrap());

pub fn infer_compute_bindgroup_layout(
    device: &wgpu::Device,
    source: &str,
    label: Option<&str>,
) -> wgpu::BindGroupLayout {
    let mut entries = Vec::new();
    for capture in SHADER_BINDGROUP_INFER.captures_iter(source) {
        let idx: u32 = capture
            .name("idx")
            .expect("Regex failed parse at binding idx")
            .as_str()
            .parse()
            .unwrap();
        let ty = capture
            .name("type")
            .expect("Regex failed parse at binding type")
            .as_str();
        let ty_err = || format!("Unrecognized symbols in binding declaration: {}", ty);
        let ty = match ty {
            "uniform" => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            _ => {
                let (storage, read_write) = ty.split_once(",").expect(&ty_err());
                let read_write = read_write.trim();
                if !(storage == "storage") {
                    panic!("{}", ty_err())
                }
                match read_write {
                    "read" => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "read_write" => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    _ => panic!("{}", ty_err()),
                }
            }
        };
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: idx,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label,
        entries: &entries[..],
    })
}

#[allow(unused_imports)]
mod tests {

    use super::*;
    use crate::shaderpreprocessor::*;
    use std::collections::HashMap;

    #[test]
    fn creation() {
        // let first = Shader::from_wgsl(include_str!("test_shaders/first.wgsl"));
        // let second = Shader::from_wgsl(include_str!("test_shaders/second.wgsl"));

        // let processor = ShaderProcessor::default();

        // let map = HashMap::<ShaderImport, Shader>::from([
        //     ("first".into(), first.into()),
        //     ("second".into(), second.into()),
        // ]);

        // let shaderdefs = [
        //     // ShaderDefVal::Bool("Hi".to_string(), false),
        //     ShaderDefVal::Any("TEST".to_string(), "Yo this is my text".to_string()),
        // ];

        // let shader = processor.process(&map[&"first".into()], &shaderdefs, &map).unwrap();

        // println!("***\n{}***", shader.get_source());
    }
}
