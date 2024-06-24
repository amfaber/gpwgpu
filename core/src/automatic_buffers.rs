use std::{
    any::{self, type_name, Any, TypeId}, collections::HashMap, fmt::Debug, hash::Hash, sync::Mutex
};

use crate::utils::{DebugEncoder, Encoder, InspectBuffer};

/// Communicates to the automatic buffer solution whether its okay for the buffer to
/// contain trash from a previous compute pass. If the pass in which is it used
/// will just write to it but not read, then its okay.
#[derive(Debug, Clone)]
pub enum MemoryReq {
    /// Its okay for this buffer to contain memory that was used by another pass
    /// previously
    UnInitOk,

    /// Ensures that the buffer is not used in a previous pass unless it had the same name
    /// in that pass.
    Strict,

    /// Implies UnInitOk and that the contents are not used later in the pipeline
    Temporary,
}

pub trait PipelineTypes: 'static {
    type Params;
    type Buffer: 'static + Hash + Eq + Clone + Copy + Debug;
    type Error;
    type Args;
}

#[derive(Debug)]
pub struct AbstractBuffer<PT: PipelineTypes> {
    pub name: PT::Buffer,
    pub memory_req: MemoryReq,
    pub usage: wgpu::BufferUsages,
    pub size: wgpu::BufferAddress,
}

// Manual clone impl to avoid a Clone bound on PipelineTypes
impl<PT: PipelineTypes> Clone for AbstractBuffer<PT>{
    fn clone(&self) -> Self {
        Self{
            name: self.name.clone(),
            memory_req: self.memory_req.clone(),
            usage: self.usage.clone(),
            size: self.size.clone(),
        }
    }
}

#[derive(Debug)]
pub struct ConcreteBuffer<PT: PipelineTypes> {
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub currently_backing: Option<(AbstractBuffer<PT>, bool)>,
}

impl<PT: PipelineTypes> ConcreteBuffer<PT> {
    fn to_wgpu_buffer(&self, device: &wgpu::Device, idx: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&idx.to_string()),
            size: self.size,
            usage: self.usage,
            mapped_at_creation: false,
        })
    }
}

fn get_or_create_buffer<PT: PipelineTypes>(
    all_buffers: &mut Vec<ConcreteBuffer<PT>>,
    ab_buf: AbstractBuffer<PT>,
    uses_for_all: wgpu::BufferUsages,
) {
    if !matches!(ab_buf.memory_req, MemoryReq::Temporary) {
        for buf in all_buffers.iter_mut() {
            let Some((cur_backing_buf, _)) = &buf.currently_backing else {
                continue;
            };
            if !matches!(cur_backing_buf.memory_req, MemoryReq::Temporary)
                && cur_backing_buf.name == ab_buf.name
            {
                buf.currently_backing = Some((ab_buf, true));
                return ();
            }
        }
    }

    if matches!(ab_buf.memory_req, MemoryReq::Strict) {
        let concrete = ConcreteBuffer {
            size: ab_buf.size,
            usage: ab_buf.usage | uses_for_all,
            currently_backing: Some((ab_buf, true)),
        };
        all_buffers.push(concrete);
        return ();
    }

    let mut indices = (0..all_buffers.len())
        .filter(|&index| {
            all_buffers[index].currently_backing.is_none()
                && all_buffers[index].size <= ab_buf.size * 10
        })
        .collect::<Vec<_>>();

    indices.sort_by_key(|&index| all_buffers[index].size);

    if let Some(&index) = indices
        .iter()
        .find(|&&index| all_buffers[index].size >= ab_buf.size)
    {
        all_buffers[index].currently_backing = Some((ab_buf, true));
    } else {
        if let Some(&index) = indices.last() {
            all_buffers[index].currently_backing = Some((ab_buf, true));
        } else {
            let concrete = ConcreteBuffer {
                size: ab_buf.size,
                usage: ab_buf.usage | uses_for_all,
                currently_backing: Some((ab_buf, true)),
            };
            all_buffers.push(concrete);
        }
    }
}

struct MapAndTypeName<PT: PipelineTypes> {
    map: HashMap<PT::Buffer, (usize, AbstractBuffer<PT>)>,
    type_name: &'static str,
}

pub struct BufferSolution<PT: PipelineTypes> {
    buffers: Vec<wgpu::Buffer>,
    assignments: HashMap<TypeId, MapAndTypeName<PT>>,
    order: Vec<TypeId>,
    pub planned_buffers: Vec<ConcreteBuffer<PT>>,
}

impl<PT: PipelineTypes> BufferSolution<PT> {
    pub fn new(
        operations: Vec<(TypeId, &'static str, Vec<AbstractBuffer<PT>>)>,
    ) -> Self {
        Self::new_internal(operations, wgpu::BufferUsages::empty())
    }

    pub fn new_dbg(
        operations: Vec<(TypeId, &'static str, Vec<AbstractBuffer<PT>>)>,
    ) -> Self {
        Self::new_internal(
            operations,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )
    }

    fn new_internal(
        operations: Vec<(TypeId, &'static str, Vec<AbstractBuffer<PT>>)>,
        uses_for_all: wgpu::BufferUsages,
    ) -> Self {
        // Representation of all the buffers we need to create
        let mut all_buffers = Vec::<ConcreteBuffer<PT>>::new();

        // For every named buffer, the index in operations that it is encountered for the last time
        let mut last_usage = HashMap::<PT::Buffer, usize>::new();

        // For every operation, a vec of the buffers that it has been assigned. u32 is the
        // binding in the bind group, while usize is an index to all_buffers.
        let mut assignments = HashMap::new();

        let mut order = Vec::new();

        for (i, (_type_id, _type_name, abs_bufs)) in operations.iter().enumerate() {
            for abs_buf in abs_bufs.iter() {
                match abs_buf.memory_req {
                    MemoryReq::UnInitOk | MemoryReq::Strict => {
                        last_usage.insert(abs_buf.name, i);
                    }
                    MemoryReq::Temporary => (),
                }
            }
        }

        for (operation_idx, (type_id, type_name, operation)) in operations.into_iter().enumerate() {
            order.push(type_id);
            for ab_buf in operation.into_iter() {
                get_or_create_buffer(&mut all_buffers, ab_buf, uses_for_all);
            }
            let mut these_assignments = HashMap::new();

            for (buf_idx, buf) in all_buffers.iter_mut().enumerate() {
                match &mut buf.currently_backing {
                    Some((abs_buf, just_set)) => {
                        if *just_set {
                            if buf.size < abs_buf.size {
                                buf.size = abs_buf.size
                            }
                            buf.usage = buf.usage.union(abs_buf.usage);
                            these_assignments.insert(abs_buf.name, (buf_idx, abs_buf.clone()));
                            *just_set = false;
                        }

                        match abs_buf.memory_req {
                            MemoryReq::UnInitOk | MemoryReq::Strict => {
                                if last_usage[&abs_buf.name] == operation_idx {
                                    buf.currently_backing = None;

                                // Purely a sanity check
                                } else if last_usage[&abs_buf.name] < operation_idx {
                                    unreachable!()
                                }
                            }
                            MemoryReq::Temporary => buf.currently_backing = None,
                        }
                    }
                    None => (),
                }
            }
            assignments.insert(
                type_id,
                MapAndTypeName {
                    map: these_assignments,
                    type_name,
                },
            );
        }

        Self {
            buffers: Vec::new(),
            assignments,
            order,
            planned_buffers: all_buffers,
        }
    }

    pub fn allocate(&mut self, device: &wgpu::Device) {
        self.buffers = self
            .planned_buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| buf.to_wgpu_buffer(device, i))
            .collect()
    }

    pub fn try_position_get(&self, operation: usize, name: PT::Buffer) -> Option<&wgpu::Buffer> {
        let id = self.order[operation];
        let &(idx, _) = self.assignments.get(&id)?.map.get(&name)?;
        Some(&self.buffers[idx])
    }

    #[track_caller]
    pub fn position_get(&self, operation: usize, name: PT::Buffer) -> &wgpu::Buffer {
        match self.try_position_get(operation, name) {
            Some(val) => val,
            None => panic!("{:?} not found in \n{:#?}\n", name, self),
        }
    }

    pub fn try_get_size<T: Any>(&self, name: PT::Buffer) -> Option<wgpu::BufferAddress> {
        let MapAndTypeName { map, type_name: _ } =
            self.assignments.get(&std::any::TypeId::of::<T>())?;
        let (_idx, abs) = map.get(&name)?;
        Some(abs.size)
    }

    #[track_caller]
    pub fn get_size<T: Any>(&self, name: PT::Buffer) -> wgpu::BufferAddress {
        match self.try_get_size::<T>(name) {
            Some(val) => val,
            None => panic!(
                "{:?} in pass {:?} not found in \n{:#?}\n",
                name,
                any::type_name::<T>(),
                self
            ),
        }
    }

    pub fn try_get<T: Any>(&self, name: PT::Buffer) -> Option<&wgpu::Buffer> {
        let MapAndTypeName { map, type_name: _ } =
            self.assignments.get(&std::any::TypeId::of::<T>())?;
        let (idx, _) = *map.get(&name)?;
        Some(&self.buffers[idx])
    }

    #[track_caller]
    pub fn get<T: Any>(&self, name: PT::Buffer) -> &wgpu::Buffer {
        match self.try_get::<T>(name) {
            Some(val) => val,
            None => panic!(
                "{:?} in pass {:?} not found in \n{:#?}\n",
                name,
                any::type_name::<T>(),
                self
            ),
        }
    }

    pub fn try_get_from_any(&self, name: PT::Buffer) -> Option<&wgpu::Buffer> {
        let idx = self
            .assignments
            .iter()
            .find_map(|(_type_id, MapAndTypeName { map, type_name: _ })| Some(map.get(&name)?.0))?;
        Some(&self.buffers[idx])
    }

    #[track_caller]
    pub fn get_from_any(&self, name: PT::Buffer) -> &wgpu::Buffer {
        match self.try_get_from_any(name) {
            Some(val) => val,
            None => panic!("{:?} not found in \n{:#?}\n", name, self),
        }
    }

    #[track_caller]
    pub fn get_bindgroup<T: Any>(&self) -> HashMap<PT::Buffer, &wgpu::Buffer> {
        match self.try_get_bindgroup::<T>() {
            Some(val) => val,
            None => panic!(
                "Pass {:?} not found in \n{:#?}\n",
                any::type_name::<T>(),
                self
            ),
        }
    }

    pub fn try_get_bindgroup<T: Any>(&self) -> Option<HashMap<PT::Buffer, &wgpu::Buffer>> {
        let MapAndTypeName { map, type_name: _ } =
            self.assignments.get(&std::any::TypeId::of::<T>())?;

        Some(
            map.iter()
                .map(|(name, (idx, _))| (name.clone(), &self.buffers[*idx]))
                .collect(),
        )
    }

    #[track_caller]
    pub fn get_inspect_buffers<T: Any>(&self) -> Vec<InspectBuffer> {
        match self.try_get_inspect_buffers::<T>() {
            Some(val) => val,
            None => panic!(
                "Pass {:?} not found in \n{:#?}\n",
                any::type_name::<T>(),
                self
            ),
        }
    }

    pub fn try_get_inspect_buffers<T: Any>(&self) -> Option<Vec<InspectBuffer>> {
        let MapAndTypeName { map, type_name: _ } =
            self.assignments.get(&std::any::TypeId::of::<T>())?;

        Some(
            map.iter()
                .map(|(name, (idx, abs))| InspectBuffer {
                    buffer: &self.buffers[*idx],
                    size: abs.size,
                    name: format!("{name:?}"),
                })
                .collect(),
        )
    }

    pub fn all_inspect_buffers(&self) -> Vec<InspectBuffer> {
        self.buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| InspectBuffer {
                buffer: buf,
                size: buf.size(),
                name: format!("ConcreteBuffer{i}"),
            })
            .collect()
    }
}

pub type PipelineParams<S> = <<S as SequentialOperation>::PT as PipelineTypes>::Params;

pub type PipelineError<S> = <<S as SequentialOperation>::PT as PipelineTypes>::Error;

pub type PipelineArgs<S> = <<S as SequentialOperation>::PT as PipelineTypes>::Args;

pub trait SequentialOperation: 'static + Debug + Any + Send + Sync {
    type PT: PipelineTypes;

    fn enabled(params: &PipelineParams<Self>) -> bool
    where
        Self: Sized;

    fn buffers(params: &PipelineParams<Self>) -> Vec<AbstractBuffer<Self::PT>>
    where
        Self: Sized;

    fn create(
        device: &wgpu::Device,
        params: &PipelineParams<Self>,
        buffers: &BufferSolution<Self::PT>,
    ) -> Result<Self, PipelineError<Self>>
    where
        Self: Sized;

    fn execute(
        &mut self,
        encoder: &mut Encoder,
        buffers: &BufferSolution<Self::PT>,
        args: &PipelineArgs<Self>,
    );

    fn set_up(
        device: &wgpu::Device,
        params: &PipelineParams<Self>,
        buffers: &BufferSolution<Self::PT>,
    ) -> Result<Box<dyn SequentialOperation<PT = Self::PT>>, PipelineError<Self>>
    where
        Self: Sized,
    {
        Ok(Box::new(Self::create(device, params, buffers)?))
    }

    fn type_id_and_buffers(
        params: &PipelineParams<Self>,
    ) -> (TypeId, &'static str, Vec<AbstractBuffer<Self::PT>>)
    where
        Self: Sized,
    {
        (
            TypeId::of::<Self>(),
            std::any::type_name::<Self>(),
            Self::buffers(params),
        )
    }

    fn my_type_id(&self) -> (TypeId, &'static str) {
        (TypeId::of::<Self>(), type_name::<Self>())
    }
}

pub struct Operation<PT: PipelineTypes> {
    enabled: fn(&PT::Params) -> bool,
    buffers: fn(&PT::Params) -> (TypeId, &'static str, Vec<AbstractBuffer<PT>>),
    set_up: fn(
        &wgpu::Device,
        &PT::Params,
        &BufferSolution<PT>,
    ) -> Result<Box<dyn SequentialOperation<PT = PT>>, PT::Error>,
}

impl<PT: PipelineTypes> Operation<PT> {
    pub fn new<S>() -> Self
    where
        S: SequentialOperation<PT = PT>,
    {
        Self {
            enabled: S::enabled,
            buffers: S::type_id_and_buffers,
            set_up: S::set_up,
        }
    }
}

pub struct AllOperations<PT: PipelineTypes> {
    pub buffers: BufferSolution<PT>,

    pub operations: Vec<Box<dyn SequentialOperation<PT = PT>>>,
    // pub operations: Mutex<Vec<Box<dyn SequentialOperation<PT = PT>>>>,

    calls: Vec<Operation<PT>>,
}

impl<PT: PipelineTypes> AllOperations<PT> {
    pub fn new(
        params: &PT::Params,
        operations: Vec<Operation<PT>>,
    ) -> Result<Self, PT::Error> {
        Self::new_internal(params, operations, false)
    }

    pub fn new_dbg(
        params: &PT::Params,
        operations: Vec<Operation<PT>>,
    ) -> Result<Self, PT::Error> {
        Self::new_internal(params, operations, true)
    }

    fn new_internal(
        params: &PT::Params,
        operations: Vec<Operation<PT>>,
        dbg: bool,
    ) -> Result<Self, PT::Error> {
        let mut all_buffers = Vec::new();
        let ops = operations
            .into_iter()
            .filter(
                |Operation {
                     enabled,
                     buffers,
                     set_up: _set_up,
                 }| {
                    let enabled = enabled(params);
                    if enabled {
                        all_buffers.push(buffers(params));
                    }
                    enabled
                },
            )
            .collect::<Vec<_>>();

        let buffers = if dbg {
            BufferSolution::new_dbg(all_buffers)
        } else {
            BufferSolution::new(all_buffers)
        };

        Ok(Self {
            buffers,
            calls: ops,
            operations: Vec::new(),
            // operations: Mutex::new(Vec::new()),
        })
    }

    pub fn reinitialize(&mut self, params: &PT::Params, dbg: bool) {
        let mut all_buffers = Vec::new();
        for Operation {
            enabled,
            buffers,
            set_up: _,
        } in self.calls.iter()
        {
            let enabled = enabled(params);
            if enabled {
                all_buffers.push(buffers(params));
            }
        }
        let buffers = if dbg {
            BufferSolution::new_dbg(all_buffers)
        } else {
            BufferSolution::new(all_buffers)
        };
        self.buffers = buffers;
    }

    pub fn finalize(
        &mut self,
        device: &wgpu::Device,
        params: &PT::Params,
    ) -> Result<(), PT::Error> {
        self.buffers.allocate(device);
        let operations = self
            .calls
            .iter()
            .map(
                |Operation {
                     enabled: _,
                     buffers: _,
                     set_up,
                 }| set_up(device, params, &self.buffers),
            )
            .collect::<Result<Vec<_>, _>>()?;

        // *self.operations.lock().unwrap() = operations;
        self.operations = operations;
        Ok(())
    }

    pub fn execute(&mut self, encoder: &mut Encoder, args: &PT::Args) {
        // let mut operations = self.operations.lock().unwrap();
        for operation in self.operations.iter_mut() {
            operation.execute(encoder, &self.buffers, args);
        }
    }

    pub fn execute_with_inspect<'a, T: 'static>(
        &'a mut self,
        encoder: &'a mut Encoder<'a>,
        args: &PT::Args,
    ) {
        // let mut operations = self.operations.lock().unwrap();
        let mut type_found = false;
        let mut type_names = Vec::new();
        encoder.activate();
        for operation in self.operations.iter_mut() {
            operation.execute(encoder, &self.buffers, args);
            let (id, name) = operation.my_type_id();
            if id == std::any::TypeId::of::<T>() {
                type_found = true;
                let DebugEncoder {
                    encoder: _,
                    debug_bundle: Some(debug_bundle),
                    debug_active,
                } = encoder
                else {
                    panic!("A debug bundle should be passed when using 'execute_with_inspect'")
                };
                if !debug_active.0 {
                    continue;
                }

                let buffers_to_inspect = self.buffers.assignments.get(&id).unwrap();
                let bufs = buffers_to_inspect
                    .map
                    .iter()
                    .map(|(name, (idx, abs))| InspectBuffer {
                        buffer: &self.buffers.buffers[*idx],
                        size: abs.size,
                        name: format!("{name:?}"),
                    })
                    .collect::<Vec<_>>();

                debug_bundle.inspects = bufs;

                let struct_name = name
                    .split("::")
                    .last()
                    .expect("The struct didn't have a name?")
                    .replace("<", "")
                    .replace(">", "");
                debug_bundle.save_path.push(struct_name);
                if !debug_bundle.save_path.exists() {
                    std::fs::create_dir(&debug_bundle.save_path)
                        .expect("The directory {file_path} could not be created");
                }

                encoder.inspect_buffers().unwrap();
            } else {
                type_names.push(name);
            }
        }

        if !type_found {
            let DebugEncoder {
                encoder: _,
                debug_bundle: Some(debug_bundle),
                debug_active: _,
            } = encoder
            else {
                panic!("A debug bundle should be passed when using 'execute_with_inspect'")
            };
            let mut new_encoder = debug_bundle
                .device
                .create_command_encoder(&Default::default());
            std::mem::swap(&mut new_encoder, encoder);
            new_encoder.finish();
            panic!(
                "Type {} not found all operations. No buffers have been dumped. All operations: {:#?}",
                std::any::type_name::<T>(),
                type_names
            );
        }
    }
}

impl<PT: PipelineTypes> Debug for MapAndTypeName<PT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries({
                let mut vec = self.map.iter().collect::<Vec<_>>();
                vec.sort_by_key(|(_, (idx, _))| idx);
                vec.into_iter().map(|(name, (index, abs))| {
                    (
                        name,
                        format!("Index: {}, Size: {:?}, {:?}", index, abs.size, abs.usage),
                    )
                })
            })
            .finish()
    }
}

impl<PT: PipelineTypes> Debug for BufferSolution<PT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.buffers.iter().enumerate().map(|(i, buf)| {
                format!(
                    "Index: {:?}, Size: {:?}, Usage: {:?}",
                    i,
                    buf.size(),
                    buf.usage()
                )
            }))
            .finish()?;
        f.write_str("\n")?;

        f.debug_map()
            .entries(self.order.iter().map(|id| {
                let map_and_type_name = self.assignments.get(id).unwrap();
                (map_and_type_name.type_name, map_and_type_name)
            }))
            .finish()
    }
}

#[test]
fn compile_check(){
    fn test<T: Send + Sync>(){
        
    }
    struct Pt;
    impl PipelineTypes for Pt{
        type Params = ();

        type Buffer = ();

        type Error = ();

        type Args = ();
    }
    test::<AllOperations<Pt>>();
}
