use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_until1},
    character::streaming::multispace0,
    combinator::{cut, map, opt},
    error::ParseError,
    number::complete::float,
    sequence::{delimited, preceded, terminated},
    IResult, Parser,
};
use wgpu::{BindGroupLayoutDescriptor, ShaderStages, StorageTextureAccess};

use crate::{
    parser::{
        get_exports, parse_tokens, process, vec_to_owned, Definition, ExpansionError,
        ExportedMoreThanOnce, NomError, Token,
    },
    utils::{Dispatcher, WorkgroupSize},
};
use pollster::FutureExt;
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fmt::Write,
    fs::DirEntry,
    path::Path,
    sync::Arc,
};

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

#[derive(thiserror::Error)]
pub enum ShaderError {
    #[error("Expansion error: {}", .0)]
    ExpansionError(#[from] ExpansionError),

    #[error("Wgpu validation error occured in this shader:\n-------------\n{}\n\n-------------\nWgpu Error: {}\n-------------", .shader, .error_string)]
    ValidationError {
        shader: String,
        error_string: String,
    },
}

impl std::fmt::Debug for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.to_string();
        f.write_str(&s)
    }
}

/// A processed [Shader]. This cannot contain preprocessor directions. It must be "ready to compile"

#[derive(Clone)]
pub struct ProcessedShader<'def> {
    pub source: String,
    pub specs: ShaderSpecs<'def>,
}

fn format_shader(shader: &str) -> String {
    let mut s = "\n".to_string();

    let n_lines = shader.lines().count() as f32;

    let pad = (n_lines.log10() + 1.0).floor() as usize;

    for (i, line) in shader.lines().enumerate() {
        write!(&mut s, "{: >width$} {line}\n", i + 1, width = pad).unwrap();
    }
    s
}

impl<'def> std::fmt::Debug for ProcessedShader<'def> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format_shader(&self.source);
        f.write_str(&s)
    }
}

impl<'def> ProcessedShader<'def> {
    pub fn get_source(&self) -> &str {
        &self.source
    }

    pub fn build(self, device: &wgpu::Device) -> Result<Arc<NonBoundPipeline>, ShaderError> {
        let Self { source, specs } = self;

        let mut bind_group_layout =
            infer_layout(&source, device, specs.bindgroup_layout_label.as_deref());

        let bind_group_layouts = bind_group_layout.iter().collect::<Vec<_>>();

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: specs.shader_label.as_deref(),
            source: wgpu::ShaderSource::Wgsl((&source).into()),
        });
        if let Some(err) = device.pop_error_scope().block_on() {
            return Err(ShaderError::ValidationError {
                error_string: err.to_string(),
                shader: format_shader(&source),
            });
        }

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelinelayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: specs.pipelinelayout_label.as_deref(),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..specs.push_constants.unwrap_or(64),
            }],
        });
        if let Some(err) = device.pop_error_scope().block_on() {
            return Err(ShaderError::ValidationError {
                error_string: err.to_string(),
                shader: format_shader(&source),
            });
        }

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: specs.pipeline_label.as_deref(),
            layout: Some(&pipelinelayout),
            module: &shader,
            entry_point: specs.entry_point.as_deref(),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(err) = device.pop_error_scope().block_on() {
            return Err(ShaderError::ValidationError {
                error_string: err.to_string(),
                shader: format_shader(&source),
            });
        }

        Ok(Arc::new(NonBoundPipeline {
            label: specs.shader_label,
            compute_pipeline,
            // Multiple binding groups are half baked right now. FullComputePass assumes 1,
            // so we just provide the first here.
            bind_group_layout: bind_group_layout.swap_remove(0),
            dispatcher: specs.dispatcher,
        }))
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
    let Some(ext) = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
    else {
        return None;
    };
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
                    Err(err) => Err(ParseShaderError {
                        name: name.clone(),
                        variant: ParseShaderErrorVariant::MultipleExports(err),
                    })
                    .unwrap(),
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
                    Err(err) => {
                        return Err(ParseShaderError {
                            name: name.to_string(),
                            variant: ParseShaderErrorVariant::MultipleExports(err),
                        })
                    }
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
}

fn attribute<'a, Error: ParseError<&'a str>>(
    attr_name: &'static str,
) -> impl Fn(&'a str) -> IResult<&'a str, u32, Error> {
    move |inp| {
        let (inp, _) = terminated(take_until("@"), tag("@"))(inp)?;

        let (inp, _) = ws(tag(attr_name))(inp)?;

        let (inp, group_idx) = delimited(ws(tag("(")), ws(float), ws(tag(")")))(inp)?;

        Ok((inp, group_idx as u32))
    }
}

// fn discard_comment<'a, F: Parser<&'a str, O, E>, O, E: ParseError<&'a str>>(
//     f: F,
// ) -> impl FnMut(&'a str) -> IResult<&'a str, O, E> {
//     preceded(
//         many0_count(tuple((
//             tag("//"),
//             take_till(|c| c == '\n'),
//             opt(char('\n')),
//         ))),
//         f,
//     )
// }

fn ws<'a, F: Parser<&'a str, O, E>, O, E: ParseError<&'a str>>(
    f: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E> {
    preceded(multispace0, f)
}

fn buffer_style(inp: &str) -> IResult<&str, wgpu::BindingType> {
    let (inp, inner) = delimited(ws(tag("<")), ws(cut(take_until1(">"))), tag(">"))(inp)?;

    let (inner, mut buffer_binding_type) = ws(alt((
        map(tag("storage"), |_| wgpu::BufferBindingType::Storage {
            read_only: true,
        }),
        map(tag("uniform"), |_| wgpu::BufferBindingType::Uniform),
    )))(inner)?;

    if let wgpu::BufferBindingType::Storage { read_only } = &mut buffer_binding_type {
        opt(preceded(
            ws(tag(",")),
            ws(map(tag("read_write"), |t| {
                *read_only = false;
                t
            })),
        ))(inner)?;
    }

    let out = wgpu::BindingType::Buffer {
        ty: buffer_binding_type,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    Ok((inp, out))
}

fn texture_style(inp: &str) -> IResult<&str, wgpu::BindingType> {
    let (inp, _) = terminated(take_until(":"), tag(":"))(inp)?;

    let (inp, ty) = ws(alt((
        map(tag("sampler"), |_| {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
        }),
        map(tag("sampler_comparison"), |_| {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)
        }),
        map(tag("texture_depth_2d"), |_| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        }),
        map(tag("texture_depth_2d_array"), |_| {
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            }
        }),
        map(tag("texture_depth_cube"), |_| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::Cube,
            multisampled: false,
        }),
        map(tag("texture_depth_cube_array"), |_| {
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::CubeArray,
                multisampled: false,
            }
        }),
        map(tag("texture_depth_multisampled_2d"), |_| {
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: true,
            }
        }),
        parse_texture_type,
    )))(inp)?;

    Ok((inp, ty))
}

fn parse_texture_type(inp: &str) -> IResult<&str, wgpu::BindingType> {
    let (inp, _) = tag("texture_")(inp)?;

    let (inp, (storage, multisampled, view_dimension)) = alt((
        map(tag("1d"), |_| {
            (false, false, wgpu::TextureViewDimension::D1)
        }),
        map(tag("storage_1d"), |_| {
            (true, false, wgpu::TextureViewDimension::D1)
        }),
        map(tag("2d"), |_| {
            (false, false, wgpu::TextureViewDimension::D2)
        }),
        map(tag("storage_2d"), |_| {
            (true, false, wgpu::TextureViewDimension::D2)
        }),
        map(tag("storage_2d_array"), |_| {
            (true, false, wgpu::TextureViewDimension::D2Array)
        }),
        map(tag("multisampled_2d"), |_| {
            (false, true, wgpu::TextureViewDimension::D2)
        }),
        map(tag("2d_array"), |_| {
            (false, false, wgpu::TextureViewDimension::D2Array)
        }),
        map(tag("3d"), |_| {
            (false, false, wgpu::TextureViewDimension::D3)
        }),
        map(tag("storage_3d"), |_| {
            (true, false, wgpu::TextureViewDimension::D3)
        }),
        map(tag("cube"), |_| {
            (false, false, wgpu::TextureViewDimension::Cube)
        }),
        map(tag("cube_array"), |_| {
            (false, false, wgpu::TextureViewDimension::CubeArray)
        }),
    ))(inp)?;

    let (inp, inner) = delimited(ws(tag("<")), ws(take_until(">")), tag(">"))(inp)?;

    let ty = if storage {
        let (inner, format) = alt((
            map(tag("rgba8unorm"), |_| wgpu::TextureFormat::Rgba8Unorm),
            map(tag("rgba8snorm"), |_| wgpu::TextureFormat::Rgba8Snorm),
            map(tag("rgba8uint"), |_| wgpu::TextureFormat::Rgba8Uint),
            map(tag("rgba8sint"), |_| wgpu::TextureFormat::Rgba8Sint),
            map(tag("rgba16uint"), |_| wgpu::TextureFormat::Rgba16Uint),
            map(tag("rgba16sint"), |_| wgpu::TextureFormat::Rgba16Sint),
            map(tag("rgba16float"), |_| wgpu::TextureFormat::Rgba16Float),
            map(tag("r32uint"), |_| wgpu::TextureFormat::R32Uint),
            map(tag("r32sint"), |_| wgpu::TextureFormat::R32Sint),
            map(tag("r32float"), |_| wgpu::TextureFormat::R32Float),
            map(tag("rg32uint"), |_| wgpu::TextureFormat::Rg32Uint),
            map(tag("rg32sint"), |_| wgpu::TextureFormat::Rg32Sint),
            map(tag("rg32float"), |_| wgpu::TextureFormat::Rg32Float),
            map(tag("rgba32uint"), |_| wgpu::TextureFormat::Rgba32Uint),
            map(tag("rgba32sint"), |_| wgpu::TextureFormat::Rgba32Sint),
            map(tag("rgba32float"), |_| wgpu::TextureFormat::Rgba32Float),
            map(tag("bgra8unorm"), |_| wgpu::TextureFormat::Bgra8Unorm),
        ))(inner)?;

        let (_, access) = preceded(
            ws(tag(",")),
            alt((
                ws(map(tag("read"), |_| StorageTextureAccess::ReadOnly)),
                ws(map(tag("write"), |_| StorageTextureAccess::WriteOnly)),
                ws(map(tag("read_write"), |_| StorageTextureAccess::ReadWrite)),
            )),
        )(inner)?;

        wgpu::BindingType::StorageTexture {
            access,
            format,
            view_dimension,
        }
    } else {
        let (_, sample_type) = alt((
            map(tag("f32"), |_| wgpu::TextureSampleType::Float {
                filterable: true,
            }),
            map(tag("i32"), |_| wgpu::TextureSampleType::Sint),
            map(tag("u32"), |_| wgpu::TextureSampleType::Uint),
        ))(inner)?;

        wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        }
    };

    Ok((inp, ty))
}

pub fn parse_layout_entry(inp: &str) -> IResult<&str, (u32, wgpu::BindGroupLayoutEntry)> {
    let (inp, group_idx) = attribute("group")(inp)?;

    let (inp, binding_idx) = attribute("binding")(inp)?;

    let (inp, _) = ws(tag("var"))(inp)?;

    let (inp, ty) = alt((buffer_style, texture_style))(inp)?;

    let out = wgpu::BindGroupLayoutEntry {
        binding: binding_idx,
        // Need to expand beyond compute at some point
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    };

    Ok((inp, (group_idx, out)))
}

/// Doesn't respect comments
pub fn infer_layout(
    mut inp: &str,
    device: &wgpu::Device,
    label: Option<&str>,
) -> Vec<wgpu::BindGroupLayout> {
    let mut map = BTreeMap::new();
    while let Ok((new_inp, (group_idx, layout))) = parse_layout_entry(inp) {
        map.entry(group_idx).or_insert(Vec::new()).push(layout);
        inp = new_inp;
    }

    map.into_iter()
        .map(|(_group_idx, entries)| {
            let desc = BindGroupLayoutDescriptor {
                label,
                entries: &entries,
            };
            let layout = device.create_bind_group_layout(&desc);
            layout
        })
        .collect()
}

#[cfg(test)]
pub mod tests {
    use crate::utils::default_device;

    use super::*;
    use pollster::FutureExt;

    #[test]
    fn yup() {
        let (device, _queue) = default_device().block_on().unwrap();
        let data = "

        // @group(0) @binding(1)
var<storage, read> buffer: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(){
    return;
}
        ";

        let _yup = infer_layout(data, &device, None);

        dbg!("Success!");
    }
}
