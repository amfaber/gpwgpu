use std::{borrow::Cow, collections::HashMap, any::{Any, TypeId, self}, cell::Cell, hash::Hash, fmt::{Display, Debug}};


/// Communicates to the automatic buffer solution whether its okay for the buffer to
/// contain trash from a previous compute pass. If the pass in which is it used
/// will just write to it but not read, then its okay.
#[derive(Debug, Clone)]
pub enum MemoryReq{
    /// Its okay for this buffer to contain memory that was used by another pass
    /// previously
    UnInitOk,

    /// Ensures that the buffer is not used in a previous pass unless it had the same name
    /// in that pass.
    Strict,
    
    /// Implies UnInitOk and that the contents are not used later in the pipeline
    Temporary,
}

#[derive(Debug, Clone)]
pub struct AbstractBuffer<B: 'static + Hash + Eq + Clone + Copy + Debug>{
    pub name: B,
    pub memory_req: MemoryReq,
    pub usage: wgpu::BufferUsages,
    pub size: wgpu::BufferAddress,
}

#[derive(Debug)]
struct ConcreteBuffer<B: 'static + Hash + Eq + Clone + Copy + Debug>{
    size: u64,
    usage: wgpu::BufferUsages,
    currently_backing: Option<(AbstractBuffer<B>, bool)>,
}

impl<B: 'static + Hash + Eq + Clone + Copy + Debug> ConcreteBuffer<B>{
    fn to_wgpu_buffer(&self, device: &wgpu::Device, idx: usize) -> wgpu::Buffer{
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&idx.to_string()),
            size: self.size,
            usage: self.usage,
            mapped_at_creation: false,
        })
    }
}

fn get_or_create_buffer<B>(
    all_buffers: &mut Vec<ConcreteBuffer<B>>,
    ab_buf: AbstractBuffer<B>,
    uses_for_all: wgpu::BufferUsages,
)
where B: 'static + Hash + Eq + Clone + Copy + Debug
{

    if !matches!(ab_buf.memory_req, MemoryReq::Temporary){
        for buf in all_buffers.iter_mut(){
            let Some((cur_backing_buf, _)) = &buf.currently_backing else { continue };
            if !matches!(cur_backing_buf.memory_req, MemoryReq::Temporary)
                && cur_backing_buf.name == ab_buf.name{
                buf.currently_backing = Some((ab_buf, true));
                return ()
            }
        }
    }

    if matches!(ab_buf.memory_req, MemoryReq::Strict){
        let concrete = ConcreteBuffer{
            size: ab_buf.size,
            usage: ab_buf.usage | uses_for_all,
            currently_backing: Some((ab_buf, true)),
        };
        all_buffers.push(concrete);
        return ()
    }

    let mut indices = (0..all_buffers.len())
        .filter(|&index| all_buffers[index].currently_backing.is_none() &&
            all_buffers[index].size <= ab_buf.size * 10)
        .collect::<Vec<_>>();
    
    indices.sort_by_key(|&index|{
        all_buffers[index].size
    });
    
    if let Some(&index) = indices.iter().find(|&&index|{
        all_buffers[index].size >= ab_buf.size
    }){
        all_buffers[index].currently_backing = Some((ab_buf, true));
    } else {
        if let Some(&index) = indices.last(){
            all_buffers[index].currently_backing = Some((ab_buf, true));
        } else {
            let concrete = ConcreteBuffer{
                size: ab_buf.size,
                usage: ab_buf.usage | uses_for_all,
                currently_backing: Some((ab_buf, true)),
            };
            all_buffers.push(concrete);
        }
    }
}

struct MapAndTypeName<B: 'static + Hash + Eq + Clone + Copy + Debug>{
    map: HashMap<B, (usize, AbstractBuffer<B>)>,
    type_name: &'static str,
}


pub struct BufferSolution<B: 'static + Hash + Eq + Clone + Copy + Debug>{
    buffers: Vec<wgpu::Buffer>,
    assignments: HashMap::<TypeId, MapAndTypeName<B>>,
    order: Vec<TypeId>,
}


impl<B: 'static + Hash + Eq + Clone + Copy + Debug> BufferSolution<B>{

    pub fn new(
        device: &wgpu::Device,
        operations: Vec<(TypeId, &'static str, Vec<AbstractBuffer<B>>)>,
    ) -> Self
    {
        Self::new_internal(device, operations, wgpu::BufferUsages::empty())
    }
    
    pub fn new_dbg(
        device: &wgpu::Device,
        operations: Vec<(TypeId, &'static str, Vec<AbstractBuffer<B>>)>,
    ) -> Self
    {
        Self::new_internal(device, operations, wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST)
    }
    
    fn new_internal(
        device: &wgpu::Device,
        operations: Vec<(TypeId, &'static str, Vec<AbstractBuffer<B>>)>,
        uses_for_all: wgpu::BufferUsages,
    ) -> Self
    {
        // Representation of all the buffers we need to create
        let mut all_buffers = Vec::<ConcreteBuffer<B>>::new();

        // For every named buffer, the index in operations that it is encountered for the last time
        let mut last_usage = HashMap::<B, usize>::new();

        // For every operation, a vec of the buffers that it has been assigned. u32 is the 
        // binding in the bind group, while usize is an index to all_buffers.
        let mut assignments = HashMap::new();

        let mut order = Vec::new();

        for (i, (_type_id, _type_name, abs_bufs)) in operations.iter().enumerate(){
            for abs_buf in abs_bufs.iter(){
                match abs_buf.memory_req{
                    MemoryReq::UnInitOk | MemoryReq::Strict => {
                        last_usage.insert(abs_buf.name, i);
                    },
                    MemoryReq::Temporary => (),
                }
            }
        }

        for (operation_idx, (type_id, type_name, operation)) in operations.into_iter().enumerate(){
            order.push(type_id);
            for ab_buf in operation.into_iter(){
                get_or_create_buffer(&mut all_buffers, ab_buf, uses_for_all);
            }
            let mut these_assignments = HashMap::new();
            
            for (buf_idx, buf) in all_buffers.iter_mut().enumerate(){
                match &mut buf.currently_backing{
                    Some((abs_buf, just_set)) => {
                        if *just_set{
                            if buf.size < abs_buf.size{
                                buf.size = abs_buf.size
                            }
                            buf.usage = buf.usage.union(abs_buf.usage);
                            these_assignments.insert(abs_buf.name, (buf_idx, abs_buf.clone()));
                            *just_set = false;
                        }

                        match abs_buf.memory_req{
                            MemoryReq::UnInitOk | MemoryReq::Strict => {
                                if last_usage[&abs_buf.name] == operation_idx{
                                    buf.currently_backing = None;

                                // Purely a sanity check
                                } else if last_usage[&abs_buf.name] < operation_idx{
                                    unreachable!()
                                }
                            },
                            MemoryReq::Temporary => {
                                buf.currently_backing = None
                            },
                        }
                    },
                    None => ()
                }
            }
            assignments.insert(type_id, MapAndTypeName { map: these_assignments, type_name });
        }
        
        let buffers = all_buffers.into_iter().enumerate().map(|(i, buf)| buf.to_wgpu_buffer(device, i)).collect();

        Self{
            buffers,
            assignments,
            order,
        }
    }

    pub fn try_position_get(&self, operation: usize, name: B) -> Option<&wgpu::Buffer>{
        let id = self.order[operation];
        let &(idx, _) = self.assignments.get(&id)?.map.get(&name)?;
        Some(&self.buffers[idx])
    }
    
    pub fn position_get(&self, operation: usize, name: B) -> &wgpu::Buffer{
        self.try_position_get(operation, name).unwrap_or_else(||{
            panic!("{:?} not found in \n{:?}\n", name, self)
        })
    }
    
    pub fn try_get_size<T: Any>(&self, name: B) -> Option<wgpu::BufferAddress>{
        let MapAndTypeName { map, type_name: _ } = self.assignments.get(&std::any::TypeId::of::<T>())?;
        let (_idx, abs) = map.get(&name)?;
        Some(abs.size)
    }
    
    pub fn get_size<T: Any>(&self, name: B) -> wgpu::BufferAddress{
        self.try_get_size::<T>(name).unwrap_or_else(||{
            panic!("{:?} in pass {:?} not found in \n{:#?}\n", name, any::type_name::<T>(), self)
        })
    }
    
    pub fn try_get<T: Any>(&self, name: B) -> Option<&wgpu::Buffer>{
        let MapAndTypeName { map, type_name: _ } = self.assignments.get(&std::any::TypeId::of::<T>())?;
        let (idx, _) = *map.get(&name)?;
        Some(&self.buffers[idx])
    }
    
    pub fn get<T: Any>(&self, name: B) -> &wgpu::Buffer{
        self.try_get::<T>(name).unwrap_or_else(||{
            panic!("{:?} in pass {:?} not found in \n{:#?}\n", name, any::type_name::<T>(), self)
        })
    }

    pub fn get_bindgroup<T: Any>(&self) -> HashMap<B, &wgpu::Buffer>{
        self.try_get_bindgroup::<T>().unwrap_or_else(||{
            panic!("Pass {:?} not found in \n{:#?}\n", any::type_name::<T>(), self)
        })
    }
    
    pub fn try_get_bindgroup<T: Any>(&self) -> Option<HashMap<B, &wgpu::Buffer>>{
        let MapAndTypeName { map, type_name: _ } = self.assignments.get(&std::any::TypeId::of::<T>())?;

        Some(map.iter().map(|(name, (idx, _))|{
            (name.clone(), &self.buffers[*idx])
        }).collect())
    }
}


type SetUpReturn<P, B, E> = Result<Box<dyn SequentialOperation<Params = P, BufferEnum = B, Error = E>>, E>;

pub trait SequentialOperation: 'static + Debug + Any{
    type Params;
    type BufferEnum: 'static + Hash + Eq + Clone + Copy + Debug;
    type Error;

    fn enabled(params: &Self::Params) -> bool where Self: Sized;
    
    fn buffers(params: &Self::Params) -> Vec<AbstractBuffer<Self::BufferEnum>> where Self: Sized;

    fn create(device: &wgpu::Device, params: &Self::Params, buffers: &BufferSolution<Self::BufferEnum>) -> Result<Self, Self::Error> where Self: Sized;
    
    fn execute(&mut self, encoder: &mut wgpu::CommandEncoder, buffers: &BufferSolution<Self::BufferEnum>);
    
    fn set_up(device: &wgpu::Device, params: &Self::Params, buffers: &BufferSolution<Self::BufferEnum>) -> SetUpReturn<Self::Params, Self::BufferEnum, Self::Error> where Self: Sized{
        Ok(Box::new(Self::create(device, params, buffers)?))
    }

    fn type_id_and_buffers(params: &Self::Params) -> (TypeId, &'static str, Vec<AbstractBuffer<Self::BufferEnum>>) where Self: Sized{
        (TypeId::of::<Self>(), std::any::type_name::<Self>(), Self::buffers(params))
    }
}

fn enabled_callback<T: SequentialOperation>() -> Box<dyn Fn(&T::Params) -> bool>{
    Box::new(T::enabled)
}

fn buffers_callback<T: SequentialOperation>() -> Box<dyn Fn(&T::Params) -> (TypeId, &'static str, Vec<AbstractBuffer<T::BufferEnum>>)>{
    Box::new(T::type_id_and_buffers)
}

fn set_up_callback<T: SequentialOperation>() -> Box<dyn Fn(&wgpu::Device, &T::Params, &BufferSolution<T::BufferEnum>) -> SetUpReturn<T::Params, T::BufferEnum, T::Error>>{
    Box::new(T::set_up)
}

pub struct Operation<P, B: 'static + Hash + Eq + Clone + Copy + Debug, E>(
    Box<dyn Fn(&P) -> bool>,
    Box<dyn Fn(&P) -> (TypeId, &'static str, Vec<AbstractBuffer<B>>)>,
    Box<dyn Fn(&wgpu::Device, &P, &BufferSolution<B>) -> SetUpReturn<P, B, E>>
);

pub fn register<T: SequentialOperation>() -> Operation<T::Params, T::BufferEnum, T::Error>{
    Operation(enabled_callback::<T>(), buffers_callback::<T>(), set_up_callback::<T>())
}

pub struct AllOperations<P, B: 'static + Hash + Eq + Clone + Copy + Debug, E>{
    pub buffers: BufferSolution<B>,

    // Cell to allow each operation to mutate their own state while having outstanding refs to
    // buffers owned by self.
    pub operations: Cell<Vec<Box<dyn SequentialOperation<Params = P, BufferEnum = B, Error = E>>>>,
}

impl<P: 'static, B: 'static + Hash + Eq + Clone + Copy + Debug, E: 'static> AllOperations<P, B, E>{
    
    pub fn new(device: &wgpu::Device, params: &P, operations: Vec<Operation<P, B, E>>) -> Result<Self, E>{
        Self::new_internal(device, params, operations, false)
    }
    
    pub fn new_dbg(device: &wgpu::Device, params: &P, operations: Vec<Operation<P, B, E>>) -> Result<Self, E>{
        Self::new_internal(device, params, operations, true)
    }


    fn new_internal(device: &wgpu::Device, params: &P, operations: Vec<Operation<P, B, E>>, dbg: bool) -> Result<Self, E>{
        let mut all_buffers = Vec::new();
        let mut set_up_callbacks = Vec::new();
        for Operation(enabled, buffers, set_up) in operations{
            if enabled(params){
                all_buffers.push(buffers(params));
                set_up_callbacks.push(set_up);
            }
        }

        let buffers = if dbg{
            BufferSolution::new_dbg(device, all_buffers)
        } else {
            BufferSolution::new(device, all_buffers)
        };

        let operations = set_up_callbacks.into_iter().map(|set_up|{
            set_up(device, params, &buffers)
        }).collect::<Result<Vec<_>, _>>()?;

        let operations = Cell::new(operations);

        Ok(Self{
            buffers,
            operations
        })
    }
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder){
        let mut operations = self.operations.take();
        for operation in operations.iter_mut(){
            operation.execute(encoder, &self.buffers);
        }
        self.operations.set(operations);
    }
}


impl<B: 'static + Hash + Eq + Clone + Copy + Debug> Debug for MapAndTypeName<B>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(
                {
                    let mut vec = self.map.iter().collect::<Vec<_>>();
                    vec.sort_by_key(|(_,(idx, _))| idx);
                    vec.into_iter().map(|(name, (index, abs))|{
                        (name,
                            format!("Index: {}, Size: {:?}, {:?}", index, abs.size, abs.usage)
                        )
                    })         
                }
            )
            .finish()
    }
}

impl<B: 'static + Hash + Eq + Clone + Copy + Debug> Debug for BufferSolution<B>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(
                self.buffers.iter().enumerate().map(|(i, buf)|{
                    format!("Index: {:?}, Size: {:?}, Usage: {:?}", i, buf.size(), buf.usage())
                })
            )
            .finish()?;
        f.write_str("\n")?;
        
        f.debug_map()
            .entries(
                self.order.iter().map(|id|{
                    let map_and_type_name = self.assignments.get(id).unwrap();
                    (map_and_type_name.type_name, map_and_type_name)
                })
            )
            .finish()
    }
}

