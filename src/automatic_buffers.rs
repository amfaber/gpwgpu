use std::{borrow::Cow, collections::{HashMap, BTreeSet}, any::{Any, TypeId}};



pub enum AbstractBufferType{
    Temporary,
    Named(Cow<'static, str>),
}

pub struct AbstractBuffer{
    ty: AbstractBufferType,
    usage: wgpu::BufferUsages,
    binding: u32,
    size: u64,
}

pub trait Operation{
    fn inputs(&self) -> Vec<AbstractBuffer>;
    fn outputs(&self) -> Vec<AbstractBuffer>;

    // fn type_name(&self) -> &'static str{
    //     std::any::type_name::<Self>()
    // }

    // fn type_name_generic() -> &'static str where Self: Sized{
    //     std::any::type_name::<Self>()
    // }
}

struct ConcreteBuffer{
    size: u64,
    usage: wgpu::BufferUsages,
    currently_backing: Option<(AbstractBuffer, bool)>,
}

impl ConcreteBuffer{
    fn to_wgpu_buffer(&self, device: &wgpu::Device, idx: usize) -> wgpu::Buffer{
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&idx.to_string()),
            size: self.size,
            usage: self.usage,
            mapped_at_creation: false,
        })
    }
}


fn get_or_create_buffer(
    all_buffers: &mut Vec<ConcreteBuffer>,
    ab_buf: AbstractBuffer,
){
    for buf in all_buffers.iter_mut(){
        match (&buf.currently_backing, &ab_buf){
            (Some((AbstractBuffer { ty: AbstractBufferType::Named(backing_name), ..}, _)),
            AbstractBuffer { ty: AbstractBufferType::Named(this_name), .. }) 
            if backing_name == this_name => {
                buf.currently_backing = Some((ab_buf, true));
                return
            }
            _ => ()
        }
    }

    
    if let Some(concrete_ref) = all_buffers.iter_mut().find(|concrete|{
        concrete.currently_backing.is_none() && concrete.size >= ab_buf.size
    }){
        concrete_ref.currently_backing = Some((ab_buf, true));
    } else {
        if let Some(concrete_ref) = all_buffers.iter_mut().rev().find(|concrete|{
            concrete.currently_backing.is_none()
        }){
            concrete_ref.currently_backing = Some((ab_buf, true));
        } else {
            let concrete = ConcreteBuffer{
                size: ab_buf.size,
                usage: ab_buf.usage,
                currently_backing: Some((ab_buf, true)),
            };
            all_buffers.push(concrete);
        }
    }
}

pub struct BufferSolution{
    buffers: Vec<wgpu::Buffer>,
    assignments: HashMap::<TypeId, Vec<(u32, usize)>>,
}


impl BufferSolution{
    pub fn new(
        device: &wgpu::Device,
        operations: Vec<&dyn Operation>,
    ) -> Self{
        // Representation of all the buffers we need to create
        let mut all_buffers = Vec::<ConcreteBuffer>::new();

        // For every named buffer, the index in operations that it is encountered for the last time
        let mut last_usage = HashMap::<Cow<'static, str>, usize>::new();

        // For every operation, a vec of the buffers that it has been assigned. u32 is the 
        // binding in the bind group, while usize is an index to all_buffers.
        let mut assignments = HashMap::<TypeId, Vec<(u32, usize)>>::new();

        for (i, operation) in operations.iter().enumerate(){
            for input in operation.inputs(){
                if let AbstractBuffer { ty: AbstractBufferType::Named(name), .. } = input{
                    last_usage.insert(name, i);
                }
            }
        }

        for operation in operations.into_iter(){
            for ab_buf in operation.inputs(){
                get_or_create_buffer(&mut all_buffers, ab_buf);
            }
            let mut these_assignments = Vec::new();
            for (i, buf) in all_buffers.iter_mut().enumerate(){
                match &mut buf.currently_backing{
                    Some((abs_buf, just_set)) => {
                        if *just_set{
                            if buf.size < abs_buf.size{
                                buf.size = abs_buf.size
                            }
                            buf.usage = buf.usage.union(abs_buf.usage);
                            these_assignments.push((abs_buf.binding, i));
                            *just_set = false;
                        }

                        if let AbstractBuffer { ty: AbstractBufferType::Named(name), .. } = abs_buf{
                            if last_usage[name] == i{
                                buf.currently_backing = None;

                            // Purely a sanity check
                            } else if last_usage[name] < i{
                                unreachable!()
                            }
                        }
                    },
                    None => ()
                }
            }
            assignments.insert(operation.type_id(), these_assignments);
        }
        
        let buffers = all_buffers.into_iter().enumerate().map(|(i, buf)| buf.to_wgpu_buffer(device, i)).collect();

        Self{
            buffers,
            assignments,
        }
    }


    fn get_bindgroup<T: Operation>(&self){
        
    }
}

