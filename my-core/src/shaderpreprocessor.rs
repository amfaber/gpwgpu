use crate::{utils::{infer_compute_bindgroup_layout, Dispatcher, WorkgroupSize}, parser::{Token, Definition, parse_tokens, process, NomError}};
use std::{
    borrow::Cow,
    collections::HashMap,
    fs::DirEntry,
    path::Path,
    rc::Rc,
};

#[derive(Debug)]
pub struct NonBoundPipeline {
    pub label: Option<String>,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub dispatcher: Option<Dispatcher>,
}

#[derive(Debug, Clone)]
pub struct ShaderSpecs<'wg, 'def> {
    pub workgroup_size: WorkgroupSize<'wg>,
    pub dispatcher: Option<Dispatcher>,
    pub push_constants: Option<u32>,
    pub shader_defs: HashMap<Cow<'def, str>, Definition<'def>>,
    pub entry_point: Option<String>,

    pub shader_label: Option<String>,
    pub bindgroup_layout_label: Option<String>,
    pub pipelinelayout_label: Option<String>,
    pub pipeline_label: Option<String>,
}

impl<'wg: 'def, 'def> ShaderSpecs<'wg, 'def> {
    pub fn new(workgroup_size: impl Into<WorkgroupSize<'wg>>) -> Self {
        let workgroup_size = workgroup_size.into();
        let shader_defs = HashMap::from([
            (workgroup_size.x_name.clone(), Definition::UInt(workgroup_size.x)),
            (workgroup_size.y_name.clone(), Definition::UInt(workgroup_size.y)),
            (workgroup_size.z_name.clone(), Definition::UInt(workgroup_size.z)),
        ]);
        Self {
            workgroup_size: workgroup_size.into(),
            dispatcher: None,
            push_constants: None,
            shader_defs,
            entry_point: None,
            shader_label: None,
            bindgroup_layout_label: None,
            pipelinelayout_label: None,
            pipeline_label: None,
        }
    }

    pub fn workgroupsize(mut self, val: WorkgroupSize<'wg>) -> Self {
        self.workgroup_size = val;
        self
    }

    pub fn dispatcher(mut self, val: Dispatcher) -> Self {
        self.dispatcher = Some(val);
        self
    }

    pub fn direct_dispatcher(mut self, dims: &[u32; 3]) -> Self {
        self.dispatcher = Some(Dispatcher::new_direct(dims, &self.workgroup_size));
        self
    }

    pub fn push_constants(mut self, val: u32) -> Self {
        self.push_constants = Some(val);
        self
    }

    pub fn extend_defs(mut self, vals: impl IntoIterator<Item = (impl Into<Cow<'wg, str>>, Definition<'def>)>) -> Self {
        let iter = vals.into_iter().map(|(key, val)| (key.into(), val));
        self.shader_defs.extend(iter);
        self
    }

    pub fn shader_label(mut self, val: &str) -> Self {
        self.shader_label = Some(val.to_string());
        self
    }

    pub fn bindgroup_layout_label(mut self, val: &str) -> Self {
        self.bindgroup_layout_label = Some(val.to_string());
        self
    }

    pub fn pipelinelayout_label(mut self, val: &str) -> Self {
        self.pipelinelayout_label = Some(val.to_string());
        self
    }

    pub fn pipeline_label(mut self, val: &str) -> Self {
        self.pipeline_label = Some(val.to_string());
        self
    }

    pub fn entry_point(mut self, val: &str) -> Self {
        self.entry_point = Some(val.to_string());
        self
    }

    pub fn labels(self, val: &str) -> Self {
        self.shader_label(val)
            .bindgroup_layout_label(val)
            .pipelinelayout_label(val)
            .pipeline_label(val)
    }
}

/// A processed [Shader]. This cannot contain preprocessor directions. It must be "ready to compile"

#[derive(Debug)]
pub struct ProcessedShader<'a, 'def> {
    pub source: String,
    pub specs: ShaderSpecs<'a, 'def>,
}

impl<'a, 'def> ProcessedShader<'a, 'def> {
    pub fn get_source(&self) -> &str {
        &self.source
    }

    pub fn build(self, device: &wgpu::Device) -> Rc<NonBoundPipeline> {
        let Self { source, specs } = self;

        let bind_group_layout = infer_compute_bindgroup_layout(
            device,
            &source,
            specs.bindgroup_layout_label.as_deref(),
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: specs.shader_label.as_deref(),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let pipelinelayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: specs.pipelinelayout_label.as_deref(),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..specs.push_constants.unwrap_or(64),
            }],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: specs.pipeline_label.as_deref(),
            layout: Some(&pipelinelayout),
            module: &shader,
            entry_point: specs.entry_point.as_deref().unwrap_or("main"),
        });

        Rc::new(NonBoundPipeline {
            label: specs.shader_label,
            compute_pipeline,
            bind_group_layout,
            dispatcher: specs.dispatcher,
        })
    }
}

#[derive(Debug)]
pub struct ParsedShader<'a>(pub Vec<Token<'a>>);



#[derive(Debug)]
pub struct ShaderProcessor<'a>(pub HashMap<&'a str, ParsedShader<'a>>);


pub fn validate_wgsl_file(
    file: std::result::Result<DirEntry, std::io::Error>,
    full_path: bool,
) -> Option<(String, String)> {
    let file = file.ok()?;
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    let path = file.path();
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase()) else { return None };
    if ext != "wgsl" {
        return None;
    }

    let name = if !full_path {
        path.file_stem().unwrap().to_os_string()
    } else {
        path.clone().into_os_string()
    };
    let name = name.into_string().ok()?;
    let path = path.into_os_string().into_string().ok()?;

    Some((name, path))
}

pub fn load_shaders_dyn(
    path: impl AsRef<Path>,
) -> Result<HashMap<String, String>, std::io::Error> {
    let read_dir = std::fs::read_dir(path)?;
    Ok(read_dir.filter_map(|file| {
        validate_wgsl_file(file, false).and_then(|(name, path)|{
            Some((name, std::fs::read_to_string(path).ok()?))
        })
    }).collect())
}

pub fn parse_shader(input: &str) -> Result<ParsedShader, NomError>{
    nom_supreme::final_parser::final_parser(parse_tokens)(input).map(ParsedShader)
}

impl<'a> ShaderProcessor<'a> {

    pub fn from_shader_hashmap(
        shaders: &'a HashMap<String, String>,
    ) -> Result<ShaderProcessor, (String, NomError)> {
        shaders.iter().map(|(name, source)| {
            Ok((name.as_str(), parse_shader(&source).map_err(|err| (name.to_string(), err))?))
        }).collect::<Result<HashMap<&str, ParsedShader>, (String, NomError)>>().map(ShaderProcessor)
    }
    
    pub fn process_by_name<'wg, 'def>(
        &self,
        name: &str,
        specs: ShaderSpecs<'wg, 'def>,
    ) -> Result<ProcessedShader<'wg, 'def>, crate::parser::ExpansionError> {
        let definitions = &specs.shader_defs;
        let lookup = |s: Cow<str>|{
            definitions.get(&s as &str).cloned()
        };
        let source = process(self.0[name].0.clone(), lookup)?;
        Ok(ProcessedShader{
            source,
            specs,
        })
    }
    
    // pub fn from_directory(
    //     &mut self,
    //     path: impl AsRef<Path>,
    //     use_full_path: bool,
    // ) -> Result<Self, std::io::Error> {
        
    //     // out.extend_from_directory(path, use_full_path)?;
    //     Ok(out)
    // }

    // pub fn extend_from_directory(
    //     &mut self,
    //     path: impl AsRef<Path>,
    //     use_full_path: bool,
    // ) -> Result<(), std::io::Error> {
    //     for file in fs::read_dir(path)? {
    //         let Some((name, file)) = Self::validate_file(file, use_full_path) else { continue };
    //         let source = Shader::from_wgsl(std::fs::read_to_string(file)?);
    //         let shader_import = if use_full_path {
    //             ShaderImport::FullPath(name)
    //         } else {
    //             ShaderImport::Name(name)
    //         };
    //         self.all_shaders.insert(shader_import, source);
    //     }
    //     Ok(())
    // }

    // pub fn from_shaders(shaders: HashMap<String, Shader>) -> Self {
    //     shaders.into()
    // }
    
}


// pub mod tests {
//     #[allow(unused_imports)]
//     use super::*;

//     #[test]
//     fn test_read_dir() {
//         // let mut map = Default::default();
//         // add_directory(&mut map, ".");
//         // dbg!(map);

//         let mut map = Default::default();
//         dbg!(_add_directory(&mut map, r"src\test_shaders", false)).unwrap();
//         dbg!(&map);
//         let _processor = ShaderProcessor::default();

//         for (_, _shader) in map.iter() {
//             // let idk = processor.process(shader, &[]);
//             // dbg!(idk.unwrap());
//         }
//         // processor.process(shader, , )
//     }
// }
