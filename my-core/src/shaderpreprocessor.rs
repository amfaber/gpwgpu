use crate::{
    parser::{
        get_exports, parse_tokens, process, vec_to_owned, Definition, ExportedMoreThanOnce,
        NomError, Token,
    },
    utils::{infer_compute_bindgroup_layout, Dispatcher, WorkgroupSize},
};
use std::{borrow::Cow, collections::HashMap, fmt::Write, fs::DirEntry, path::Path, rc::Rc};

#[derive(Debug)]
pub struct NonBoundPipeline {
    pub label: Option<String>,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub dispatcher: Option<Dispatcher<'static>>,
}

#[derive(Debug, Clone)]
pub struct ShaderSpecs<'def> {
    pub workgroup_size: WorkgroupSize,
    pub dispatcher: Option<Dispatcher<'static>>,
    pub push_constants: Option<u32>,
    pub shader_defs: HashMap<Cow<'def, str>, Definition<'def>>,
    pub entry_point: Option<String>,

    pub shader_label: Option<String>,
    pub bindgroup_layout_label: Option<String>,
    pub pipelinelayout_label: Option<String>,
    pub pipeline_label: Option<String>,
}

impl<'def> ShaderSpecs<'def> {
    pub fn new(workgroup_size: impl Into<WorkgroupSize>) -> Self {
        let workgroup_size = workgroup_size.into();
        let shader_defs = HashMap::from([
            (
                workgroup_size.x_name.clone(),
                Definition::UInt(workgroup_size.x),
            ),
            (
                workgroup_size.y_name.clone(),
                Definition::UInt(workgroup_size.y),
            ),
            (
                workgroup_size.z_name.clone(),
                Definition::UInt(workgroup_size.z),
            ),
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

    pub fn workgroupsize(mut self, val: WorkgroupSize) -> Self {
        self.workgroup_size = val;
        self
    }

    pub fn dispatcher(mut self, val: Dispatcher<'static>) -> Self {
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

    pub fn extend_defs(
        mut self,
        vals: impl IntoIterator<Item = (impl Into<Cow<'static, str>>, Definition<'def>)>,
    ) -> Self {
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

#[derive(Debug, thiserror::Error)]
enum ParseShaderErrorVariant<'a> {
    #[error("{}", .0)]
    MultipleExports(#[from] ExportedMoreThanOnce),
    #[error("{}", .0)]
    NomError(NomError<'a>),
}

#[derive(Debug, thiserror::Error)]
#[error("Parsing the shader {} encountered an error: {}", .name, .variant)]
pub struct ParseShaderError<'a> {
    name: String,
    variant: ParseShaderErrorVariant<'a>,
}

/// A processed [Shader]. This cannot contain preprocessor directions. It must be "ready to compile"

#[derive(Clone)]
pub struct ProcessedShader<'def> {
    pub source: String,
    pub specs: ShaderSpecs<'def>,
}

impl<'def> std::fmt::Debug for ProcessedShader<'def> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = "\n".to_string();

        let n_lines = self.source.lines().count() as f32;

        let pad = (n_lines.log10() + 1.0).floor() as usize;

        for (i, line) in self.source.lines().enumerate() {
            write!(&mut s, "{: >width$} {line}\n", i + 1, width = pad).unwrap();
        }

        f.write_str(&s)
    }
}

impl<'def> ProcessedShader<'def> {
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

impl<'a> ParsedShader<'a> {
    fn into_owned(self) -> ParsedShader<'static> {
        ParsedShader(vec_to_owned(self.0))
    }

    fn get_exports(
        &self,
        exports: &mut HashMap<Cow<'a, str>, Vec<Token<'a>>>,
    ) -> Result<(), ExportedMoreThanOnce> {
        get_exports(&self.0, exports)
    }
}

#[derive(Debug)]
pub struct ShaderProcessor<'a> {
    pub shaders: HashMap<Cow<'a, str>, ParsedShader<'a>>,
    pub exports: HashMap<Cow<'a, str>, Vec<Token<'a>>>,
}

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

pub fn parse_shader<'a>(input: &'a str) -> Result<ParsedShader<'a>, NomError<'a>> {
    nom_supreme::final_parser::final_parser(parse_tokens)(input).map(ParsedShader)
}

impl<'a> ShaderProcessor<'a> {
    pub fn load_dir_dyn(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let read_dir = std::fs::read_dir(path)?;

        let mut exports = HashMap::new();

        let shaders = read_dir
            .filter_map(|file| {
                validate_wgsl_file(file, false).and_then(|(name, path)| {
                    let source = std::fs::read_to_string(path).ok()?;
                    Some((name, source))
                })
            })
            .map(|(name, source)| {
                let parsed = match parse_shader(&source) {
                    Ok(val) => val,
                    Err(e) => {
                        println!("Failed parsing of shader {name}");
                        panic!("{}", e);
                    }
                }
                .into_owned();
                // FIXME probably shouldn't unwrap here. Constructing the right error atleast
                // gives some formatting. Can't quite decide how to deal with this shader
                // level error + folder level io error.
                match parsed.get_exports(&mut exports) {
                    Ok(()) => (),
                    Err(err) => Err(ParseShaderError{
                        name: name.clone(),
                        variant: ParseShaderErrorVariant::MultipleExports(err),
                    }).unwrap(),
                };
                (name.into(), parsed)
            })
            .collect();
        Ok(Self { shaders, exports })
    }

    pub fn from_shader_hashmap(
        shaders: &'a HashMap<Cow<'a, str>, String>,
    ) -> Result<ShaderProcessor<'a>, ParseShaderError<'a>> {
        let mut exports = HashMap::new();
        let shaders = shaders
            .iter()
            .map(|(name, source)| {
                let parsed = parse_shader(&source).map_err(|err| ParseShaderError {
                    name: name.to_string(),
                    variant: ParseShaderErrorVariant::NomError(err),
                })?;
                match parsed.get_exports(&mut exports) {
                    Ok(()) => (),
                    Err(err) => return Err(ParseShaderError{
                        name: name.to_string(),
                        variant: ParseShaderErrorVariant::MultipleExports(err),
                    })
                };
                Ok((name.clone(), parsed))
            })
            .collect::<Result<HashMap<Cow<str>, ParsedShader>, ParseShaderError>>()?;

        Ok(Self { shaders, exports })
    }

    pub fn from_parsed_shader_hashmap(
        shaders: HashMap<Cow<'a, str>, ParsedShader<'a>>,
    ) -> Result<ShaderProcessor, ParseShaderError> {
        let mut exports = HashMap::new();
        for (name, parsed) in shaders.iter() {
            match parsed.get_exports(&mut exports) {
                Ok(()) => (),
                Err(e) => {
                    return Err(ParseShaderError {
                        name: name.to_string(),
                        variant: e.into(),
                    })
                }
            }
        }

        Ok(Self { shaders, exports })
    }

    pub fn process_by_name<'wg, 'def>(
        &self,
        name: &str,
        specs: ShaderSpecs<'def>,
    ) -> Result<ProcessedShader<'def>, crate::parser::ExpansionError> {
        let definitions = &specs.shader_defs;
        let lookup = |s: Cow<str>| definitions.get(&s as &str).cloned();
        let exports = |s| self.exports.get(&s).cloned();
        let source = process(self.shaders[name].0.clone(), lookup, exports)?;
        Ok(ProcessedShader { source, specs })
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
