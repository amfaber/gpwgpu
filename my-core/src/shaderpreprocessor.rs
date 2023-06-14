use std::{borrow::Cow, collections::HashMap, path::Path, fs::{self, DirEntry}, rc::Rc, num::ParseIntError};
use thiserror::Error;
use regex::Regex;
use once_cell::sync::Lazy;
use crate::utils::{Dispatcher, infer_compute_bindgroup_layout, WorkgroupSize};

#[derive(Clone, PartialEq, Debug)]
// #[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum ShaderDefVal {
    Bool(String, bool),
    Int(String, i32),
    UInt(String, u32),
    Any(String, String),
    Float(String, f32),
    IRange(String, std::ops::Range<i32>),
    URange(String, std::ops::Range<u32>),
}

impl std::hash::Hash for ShaderDefVal{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self{
            ShaderDefVal::Bool(name, val) => {name.hash(state); val.hash(state)},
            ShaderDefVal::Int(name, val) => {name.hash(state); val.hash(state)},
            ShaderDefVal::UInt(name, val) => {name.hash(state); val.hash(state)},
            ShaderDefVal::Float(name, val) => {name.hash(state); val.to_bits().hash(state)},
            ShaderDefVal::Any(name, val) => {name.hash(state); val.hash(state)},
            ShaderDefVal::IRange(name, val) => {name.hash(state); val.hash(state)},
            ShaderDefVal::URange(name, val) => {name.hash(state); val.hash(state)},
        }
    }
}


impl From<&str> for ShaderDefVal {
    fn from(key: &str) -> Self {
        ShaderDefVal::Bool(key.to_string(), true)
    }
}

impl From<String> for ShaderDefVal {
    fn from(key: String) -> Self {
        ShaderDefVal::Bool(key, true)
    }
}

impl ShaderDefVal {
    pub fn value_as_string(&self) -> Option<String> {
        match self {
            ShaderDefVal::Bool(_, def) => Some(def.to_string()),
            ShaderDefVal::Int(_, def) => Some(def.to_string()),
            ShaderDefVal::UInt(_, def) => {
                let mut out = def.to_string();
                out.push('u');
                Some(out)
            },
            ShaderDefVal::Float(_, def) => Some(def.to_string()),
            ShaderDefVal::Any(_, def) => Some(def.to_string()),
            
            ShaderDefVal::IRange(_, _) => None,
            ShaderDefVal::URange(_, _) => None,
        }
    }
}


#[derive(Debug, Clone)]
// #[uuid = "d95bc916-6c55-4de3-9622-37e7b6969fda"]
pub struct Shader {
    source: Source,
    import_path: Option<ShaderImport>,
    imports: Vec<ShaderImport>,
}

impl Shader {
    pub fn from_wgsl(source: impl Into<Cow<'static, str>>) -> Shader {
        let source = source.into();
        let shader_imports = SHADER_IMPORT_PROCESSOR.get_imports_from_str(&source);
        Shader {
            imports: shader_imports.imports,
            import_path: shader_imports.import_path,
            source: Source(source),
        }
    }

    pub fn set_import_path<P: Into<String>>(&mut self, import_path: P) {
        self.import_path = Some(ShaderImport::FullPath(import_path.into()));
    }

    #[must_use]
    pub fn with_import_path<P: Into<String>>(mut self, import_path: P) -> Self {
        self.set_import_path(import_path);
        self
    }

    #[inline]
    pub fn import_path(&self) -> Option<&ShaderImport> {
        self.import_path.as_ref()
    }

    pub fn imports(&self) -> impl ExactSizeIterator<Item = &ShaderImport> {
        self.imports.iter()
    }
}

#[derive(Debug, Clone)]
pub struct Source(Cow<'static, str>);

/// A processed [Shader]. This cannot contain preprocessor directions. It must be "ready to compile"
// #[derive(PartialEq, Eq, Debug, Clone)]
#[derive(Debug)]
pub struct ProcessedShader{
    pub source: String,
    pub specs: ShaderSpecs,
}

impl ProcessedShader {
    pub fn get_source(&self) -> &str {
		&self.source
    }

    pub fn build(self, device: &wgpu::Device) -> Rc<NonBoundPipeline>{
        let Self{source, specs} = self;
        
        let bind_group_layout = infer_compute_bindgroup_layout(device, &source, specs.bindgroup_layout_label.as_deref());
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: specs.shader_label.as_deref(),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let pipelinelayout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: specs.pipelinelayout_label.as_deref(),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange{
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..specs.push_constants.unwrap_or(64),
            }],
        });
        
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: specs.pipeline_label.as_deref(),
            layout: Some(&pipelinelayout),
            module: &shader,
            entry_point: specs.entry_point.as_deref().unwrap_or("main"),
        });
        
        Rc::new(NonBoundPipeline{
            label: specs.shader_label,
            compute_pipeline,
            bind_group_layout,
            dispatcher: specs.dispatcher,
        })
    }
    
}

#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum ProcessShaderError {
    #[error("Too many '# endif' lines. Each endif should be preceded by an if statement.")]
    TooManyEndIfs,
    
    #[error("Too many '# endfor' lines. Each endfor should be preceded by an for statement.")]
    TooManyEndFors,
    
    #[error(
        "Not enough '# endif' lines. Each if statement should be followed by an endif statement."
    )]
    NotEnoughEndIfs,
    
    #[error(
        "Not enough '# endfor' lines. Each for statement should be followed by an endfor statement."
    )]
    NotEnoughEndFors,
    
    #[error("This Shader's format does not support processing shader defs.")]
    ShaderFormatDoesNotSupportShaderDefs,
    
    #[error("This Shader's format does not support imports.")]
    ShaderFormatDoesNotSupportImports,
    
    #[error("Unresolved import: {0:?}.")]
    UnresolvedImport(ShaderImport),
    
    #[error("The shader import {0:?} does not match the source file type. Support for this might be added in the future.")]
    MismatchedImportFormat(ShaderImport),
    
    #[error("Unknown shader def operator: '{operator}'")]
    UnknownShaderDefOperator { operator: String },
    
    #[error("Unknown shader def: '{shader_def_name}'")]
    UnknownShaderDef { shader_def_name: String },
    
    #[error(
        "Invalid shader def comparison for '{shader_def_name}': expected {expected}, got {value}"
    )]
    InvalidShaderDefComparisonValue {
        shader_def_name: String,
        expected: String,
        value: String,
    },
    
    #[error(
        "Invalid shader def comparison for '{shader_def_name}' with value '{value}': Only != and == are allowed for string comparisons"
    )]
    InvalidShaderDefComparisonAny {
        shader_def_name: String,
        value: String,
    },
    
    #[error("Invalid shader def definition for '{shader_def_name}': {value}")]
    InvalidShaderDefDefinitionValue {
        shader_def_name: String,
        value: String,
    },

    #[error("Invalid Key")]
    InvalidKey{
        key: String
    },

    #[error("Failed to parse 'for' range")]
    InvalidForRange{
        ident: String,
        parse_error: ParseIntError,
    },

    #[error("No range provided for 'for' loop")]
    NoRangeForLoop{
        ident: String,
    }
}

pub struct ShaderImportProcessor {
    import_name_regex: Regex,
    import_full_path_regex: Regex,
    define_import_path_regex: Regex,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ShaderImport {
    Name(String),
    FullPath(String),
}

impl From<&str> for ShaderImport{
    fn from(value: &str) -> Self {
        Self::Name(value.to_string())
    }
}

impl From<String> for ShaderImport{
    fn from(value: String) -> Self {
        Self::Name(value)
    }
}

impl From<&Self> for ShaderImport{
    fn from(value: &Self) -> Self {
        value.clone()
    }
}

impl ShaderImport{
    fn to_string(self) -> String{
        match self{
            Self::Name(str) => str,
            Self::FullPath(str) => str,
        }
    }
}

impl Default for ShaderImportProcessor {
    fn default() -> Self {
        Self {
            import_name_regex: Regex::new(r"^\s*#\s*import\s+(.+)").unwrap(),
            import_full_path_regex: Regex::new(r#"^\s*#\s*import\s+"(.+)""#).unwrap(),
            define_import_path_regex: Regex::new(r"^\s*#\s*define_import_path\s+(.+)").unwrap(),
        }
    }
}

#[derive(Default)]
pub struct ShaderImports {
    imports: Vec<ShaderImport>,
    import_path: Option<ShaderImport>,
}

impl ShaderImportProcessor {
    pub fn get_imports(&self, shader: &Shader) -> ShaderImports {
		self.get_imports_from_str(&shader.source.0)
        // match &shader.source {
        //     Source::Wgsl(source) => self.get_imports_from_str(source),
        //     Source::Glsl(source, _stage) => self.get_imports_from_str(source),
        //     Source::SpirV(_source) => ShaderImports::default(),
        // }
    }

    pub fn get_imports_from_str(&self, shader: &str) -> ShaderImports {
        let mut shader_imports = ShaderImports::default();
        for line in shader.lines() {
            if let Some(cap) = self.import_name_regex.captures(line) {
                let import = cap.get(1).unwrap();
                shader_imports
                    .imports
                    .push(ShaderImport::Name(import.as_str().to_string()));
            } else if let Some(cap) = self.import_full_path_regex.captures(line) {
                let import = cap.get(1).unwrap();
                shader_imports
                    .imports
                    .push(ShaderImport::FullPath(import.as_str().to_string()));
            } else if let Some(cap) = self.define_import_path_regex.captures(line) {
                let path = cap.get(1).unwrap();
                shader_imports.import_path = Some(ShaderImport::FullPath(path.as_str().to_string()));
            }
        }

        shader_imports
    }
}

pub static SHADER_IMPORT_PROCESSOR: Lazy<ShaderImportProcessor> =
    Lazy::new(ShaderImportProcessor::default);

pub struct ShaderProcessor {
    ifdef_regex: Regex,
    ifndef_regex: Regex,
    ifop_regex: Regex,
    else_ifdef_regex: Regex,
    else_regex: Regex,
    endif_regex: Regex,
    define_regex: Regex,
    def_regex: Regex,
    def_regex_delimited: Regex,

    for_regex: Regex,
    endfor_regex: Regex,

    pub all_shaders: HashMap<ShaderImport, Shader>,
}

impl Default for ShaderProcessor {
    fn default() -> Self {
        Self {
            ifdef_regex: Regex::new(r"^\s*#\s*ifdef\s*([\w|\d|_]+)").unwrap(),
            ifndef_regex: Regex::new(r"^\s*#\s*ifndef\s*([\w|\d|_]+)").unwrap(),
            ifop_regex: Regex::new(r"^\s*#\s*if\s*([\w|\d|_]+)\s*([^\s]*)\s*([-\w|\d]+)").unwrap(),
            else_ifdef_regex: Regex::new(r"^\s*#\s*else\s+ifdef\s*([\w|\d|_]+)").unwrap(),
            else_regex: Regex::new(r"^\s*#\s*else").unwrap(),
            endif_regex: Regex::new(r"^\s*#\s*endif").unwrap(),
            define_regex: Regex::new(r"^\s*#\s*define\s+(\w+)\s*(-?\w+\.\d+|-?\w+)?").unwrap(),
            def_regex: Regex::new(r"#\s*([\w|\d|_]+)").unwrap(),
            def_regex_delimited: Regex::new(r"#\s*\{([\w|\d|_]+)\}").unwrap(),
            
            for_regex: Regex::new(r"^\s*#\s*for\s*([\w|\d|_]+)(?:\s+in\s+([-\w|\d]+)\.\.([-\w|\d]+))?").unwrap(),
            endfor_regex: Regex::new(r"^\s*#\s*endfor").unwrap(),

            all_shaders: HashMap::new(),
        }
    }
}

#[derive(Clone, Copy)]
enum MaybeSigned{
    Signed(i32),
    Unsigned(u32),
}

impl MaybeSigned{
    fn to_signed(self) -> i32{
        match self{
            Self::Signed(val) => val,
            Self::Unsigned(val) => val as i32,
        }
    }
}


fn parse_side(
    side: &str,
    shader_defs_unique: &HashMap<String, ShaderDefVal>
) -> Result<MaybeSigned, ParseIntError>{
    
    let (key, is_neg) = if side.get(0..1).unwrap() == "-"{
        (side.get(1..).unwrap(), true)
    } else {
        (side, false)
    };
    let val = match shader_defs_unique.get(key){
        Some(ShaderDefVal::UInt(_, val)) => {
            if is_neg{
                MaybeSigned::Signed(-(*val as i32))
            } else {
                MaybeSigned::Unsigned(*val)
            }
        },
        Some(ShaderDefVal::Int(_, val)) => {
            if is_neg{
                MaybeSigned::Signed(-(*val))
            } else {
                MaybeSigned::Signed(*val)
            }
        },
        _ => {
            if side.get(side.len() - 1..side.len()).unwrap() == "u"{
                MaybeSigned::Unsigned(side[..side.len() - 1].parse::<u32>()?)
            } else {
                MaybeSigned::Signed(side.parse::<i32>()?)
            }
        }
    };
    
    Ok(val)
}

fn range_from_str(
    ident: &str,
    start: &str,
    end: &str,
    shader_defs_unique: &HashMap<String, ShaderDefVal>
) -> Result<ShaderDefVal, ParseIntError>{
        
    let start = parse_side(start, shader_defs_unique)?;
    let end = parse_side(end, shader_defs_unique)?;

    let range = match (start, end){
        (MaybeSigned::Unsigned(start), MaybeSigned::Unsigned(end)) => ShaderDefVal::URange(ident.to_string(), start..end),
        _ => ShaderDefVal::IRange(ident.to_string(), start.to_signed()..end.to_signed()),
    };
    
    Ok(range)
}


impl ShaderProcessor {
    pub fn validate_file(
        file: std::result::Result<DirEntry, std::io::Error>,
        full_path: bool
    ) -> Option<(String, String)>{
        let file = file.ok()?;
        if !file.metadata().ok()?.is_file(){
            return None
        }
        let path = file.path();
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase()) else { return None };
        if ext != "wgsl"{
            return None
        }

        let name = if !full_path{
            path.file_stem().unwrap().to_os_string()
        } else {
            path.clone().into_os_string()
        };
        let name = name.into_string().ok()?;
        let path = path.into_os_string().into_string().ok()?;
    
        Some((name, path))
    }


    pub fn from_directory(&mut self,
        path: impl AsRef<Path>,
        use_full_path: bool
    ) -> Result<Self, std::io::Error>{
        let mut out = Self::default();
        out.extend_from_directory(path, use_full_path)?;
        Ok(out)
    }

    pub fn extend_from_directory(&mut self,
        path: impl AsRef<Path>,
        use_full_path: bool
    ) -> Result<(), std::io::Error>{
        for file in fs::read_dir(path)?{
            let Some((name, file)) = Self::validate_file(file, use_full_path) else { continue };
            let source = Shader::from_wgsl(std::fs::read_to_string(file)?);
            let shader_import = if use_full_path{
                ShaderImport::FullPath(name)
            } else {
                ShaderImport::Name(name)
            };
            self.all_shaders.insert(shader_import, source);
        }
        Ok(())
    }
    
    pub fn from_shaders(shaders: HashMap<ShaderImport, Shader>) -> Self{
        shaders.into()
    }
    
    pub fn process_by_key<K: Into<ShaderImport>>(
        &self,
        key: K,
        specs: ShaderSpecs,
    ) -> Result<ProcessedShader, ProcessShaderError>{
        let shader_import = key.into();
        let shader = self.all_shaders.get(&shader_import).ok_or_else(||{
            ProcessShaderError::InvalidKey { key: shader_import.clone().to_string() }
        })?;
        self.process(shader, specs)
    }
    
    pub fn process(
        &self,
        shader: &Shader,
        specs: ShaderSpecs,
    ) -> Result<ProcessedShader, ProcessShaderError> {
        let mut shader_defs_unique =
            HashMap::<String, ShaderDefVal>::from_iter(specs.shader_defs.iter().rev().map(|v| match v {
                ShaderDefVal::Bool(k, _)
                | ShaderDefVal::Int(k, _)
                | ShaderDefVal::UInt(k, _)
                | ShaderDefVal::Float(k, _)
                | ShaderDefVal::IRange(k, _)
                | ShaderDefVal::URange(k, _)
                | ShaderDefVal::Any(k, _) => {
                    (k.clone(), v.clone())
                }
            }));
        let source = self.process_inner(shader, &mut shader_defs_unique)?;
        Ok(ProcessedShader { source, specs })
    }
    
    fn process_inner<'a>(
        &self,
        shader: &Shader,
        shader_defs_unique: &mut HashMap<String, ShaderDefVal>,
    ) -> Result<String, ProcessShaderError> {
        
        let shader_str: &str = &shader.source.0;
        let mut scopes = vec![IfScope::new(true)];
        let mut for_scopes = vec![ForScope{ ident: String::new(), string: String::new() }];
        let for_scopes_raw = &for_scopes as *const Vec<ForScope>;
        let for_scopes_len = || unsafe{(*for_scopes_raw).len()} ;
        let mut write_to = &mut for_scopes.last_mut().unwrap().string;
        // let mut dummy = String::new();
        for line in shader_str.lines() {
            if let Some(start) = &line.trim_start().get(0..2){
                if start == &"//"{
                    write_to.push_str(line);
                    write_to.push('\n');
                    continue
                }
            }

            if let Some(cap) = self.for_regex.captures(line){
                let ident = cap.get(1).unwrap().as_str().to_string();

                if let (Some(start), Some(end)) = (cap.get(2), cap.get(3)){
                    let range = range_from_str(&ident, start.as_str(), end.as_str(), &shader_defs_unique)
                        .map_err(|err|{
                            ProcessShaderError::InvalidForRange { ident: ident.clone(), parse_error: err }
                        })?;
                    shader_defs_unique.insert(ident.clone(), range);
                } else {
                    if !shader_defs_unique.contains_key(&ident){
                        return Err(ProcessShaderError::NoRangeForLoop { ident: ident.clone() })
                    };
                }
                for_scopes.push(ForScope{
                    ident,
                    string: String::new(),
                });
                write_to = &mut for_scopes.last_mut().unwrap().string;
            } else if self.endfor_regex.is_match(line){
                if for_scopes_len() < 2{
                    return Err(ProcessShaderError::TooManyEndFors)
                }

                let ForScope{ ident, string } = for_scopes.pop().unwrap();

                let range = shader_defs_unique.get(&ident).expect("A range should always be present at this point");
                
                let mut full = String::new();
                match range{
                    ShaderDefVal::URange(ident, range) => {
                        for i in range.clone(){
                            let mut number = i.to_string();
                            number.push('u');
                            let string = string.replace(&format!("#{}", ident),
                                &number);
                            full.push_str(&string);
                        }
                    },
                    ShaderDefVal::IRange(ident, range) => {
                        for i in range.clone(){
                            let string = string.replace(&format!("#{}", ident),
                                &i.to_string());
                            full.push_str(&string);
                        }
                    },
                    _ => panic!("For loop identifiers should always be ranges")
                };
                
                write_to = &mut for_scopes.last_mut().unwrap().string;
                write_to.push_str(&full);
                
            } else if let Some(cap) = self.ifdef_regex.captures(line) {
                let def = cap.get(1).unwrap();

                let current_valid = scopes.last().unwrap().is_accepting_lines();
                let has_define = shader_defs_unique.contains_key(def.as_str());

                scopes.push(IfScope::new(current_valid && has_define));
            } else if let Some(cap) = self.ifndef_regex.captures(line) {
                let def = cap.get(1).unwrap();

                let current_valid = scopes.last().unwrap().is_accepting_lines();
                let has_define = shader_defs_unique.contains_key(def.as_str());

                scopes.push(IfScope::new(current_valid && !has_define));
            } else if let Some(cap) = self.ifop_regex.captures(line) {
                let def = cap.get(1).unwrap();
                let op = cap.get(2).unwrap();
                let val = cap.get(3).unwrap();

                fn act_on<T: PartialEq + PartialOrd>(a: T, b: T, op: &str) -> Result<bool, ProcessShaderError> {
                    match op {
                        "==" => Ok(a == b),
                        "!=" => Ok(a != b),
                        ">" => Ok(a > b),
                        ">=" => Ok(a >= b),
                        "<" => Ok(a < b),
                        "<=" => Ok(a <= b),
                        _ => Err(ProcessShaderError::UnknownShaderDefOperator {
                            operator: op.to_string(),
                        }),
                    }
                }

                let def = shader_defs_unique.get(def.as_str()).ok_or(
                    ProcessShaderError::UnknownShaderDef {
                        shader_def_name: def.as_str().to_string(),
                    },
                )?;
                let new_scope = match def {
                    ShaderDefVal::Bool(name, def) => {
                        let val = val.as_str().parse().map_err(|_| {
                            ProcessShaderError::InvalidShaderDefComparisonValue {
                                shader_def_name: name.clone(),
                                value: val.as_str().to_string(),
                                expected: "bool".to_string(),
                            }
                        })?;
                        act_on(*def, val, op.as_str())?
                    }
                    ShaderDefVal::Int(name, def) => {
                        let val = val.as_str().parse().map_err(|_| {
                            ProcessShaderError::InvalidShaderDefComparisonValue {
                                shader_def_name: name.clone(),
                                value: val.as_str().to_string(),
                                expected: "int".to_string(),
                            }
                        })?;
                        act_on(*def, val, op.as_str())?
                    }
                    ShaderDefVal::UInt(name, def) => {
                        let val = val.as_str().parse().map_err(|_| {
                            ProcessShaderError::InvalidShaderDefComparisonValue {
                                shader_def_name: name.clone(),
                                value: val.as_str().to_string(),
                                expected: "uint".to_string(),
                            }
                        })?;
                        act_on(*def, val, op.as_str())?
                    }
                    ShaderDefVal::Float(name, def) => {
                        let val = val.as_str().parse().map_err(|_| {
                            ProcessShaderError::InvalidShaderDefComparisonValue {
                                shader_def_name: name.clone(),
                                value: val.as_str().to_string(),
                                expected: "float".to_string(),
                            }
                        })?;
                        act_on(*def, val, op.as_str())?
                    }
                    ShaderDefVal::Any(name, def) => {
                        let op_str = op.as_str();
                        if !((op_str == "==") | (op_str == "!=")){
                            return Err(ProcessShaderError::InvalidShaderDefComparisonAny{
                                shader_def_name: name.to_string(),
                                value: def.as_str().to_string(),
                            })
                        }
                        act_on(def.as_str(), val.as_str(), op_str)?
                    }
                    ShaderDefVal::IRange(_, _) | ShaderDefVal::URange(_, _) => {
                        panic!("Inteactions between 'for' and 'if' not yet implemented")
                    }
                };

                let current_valid = scopes.last().unwrap().is_accepting_lines();

                scopes.push(IfScope::new(current_valid && new_scope));
            } else if let Some(cap) = self.else_ifdef_regex.captures(line) {
                // When should we accept the code in an
                //
                //  #else ifdef FOO
                //      <stuff>
                //  #endif
                //
                // block? Conditions:
                //  1. The parent scope is accepting lines.
                //  2. The current scope is _not_ accepting lines.
                //  3. FOO is defined.
                //  4. We haven't already accepted another #ifdef (or #else ifdef) in the current scope.

                // Condition 1
                let mut parent_accepting = true;

                if scopes.len() > 1 {
                    parent_accepting = scopes[scopes.len() - 2].is_accepting_lines();
                }

                if let Some(current) = scopes.last_mut() {
                    // Condition 2
                    let current_accepting = current.is_accepting_lines();

                    // Condition 3
                    let def = cap.get(1).unwrap();
                    let has_define = shader_defs_unique.contains_key(def.as_str());

                    if parent_accepting && !current_accepting && has_define {
                        // Condition 4: Enforced by [`Scope`].
                        current.start_accepting_lines_if_appropriate();
                    } else {
                        current.stop_accepting_lines();
                    }
                }
            } else if self.else_regex.is_match(line) {
                let mut parent_accepting = true;

                if scopes.len() > 1 {
                    parent_accepting = scopes[scopes.len() - 2].is_accepting_lines();
                }
                if let Some(current) = scopes.last_mut() {
                    // Using #else means that we only want to accept those lines in the output
                    // if the stuff before #else was _not_ accepted.
                    // That's why we stop accepting here if we were currently accepting.
                    //
                    // Why do we care about the parent scope?
                    // Because if we have something like this:
                    //
                    //  #ifdef NOT_DEFINED
                    //      // Not accepting lines
                    //      #ifdef NOT_DEFINED_EITHER
                    //          // Not accepting lines
                    //      #else
                    //          // This is now accepting lines relative to NOT_DEFINED_EITHER
                    //          <stuff>
                    //      #endif
                    //  #endif
                    //
                    // We don't want to actually add <stuff>.

                    if current.is_accepting_lines() || !parent_accepting {
                        current.stop_accepting_lines();
                    } else {
                        current.start_accepting_lines_if_appropriate();
                    }
                }
            } else if self.endif_regex.is_match(line) {
                scopes.pop();
                if scopes.is_empty() {
                    return Err(ProcessShaderError::TooManyEndIfs);
                }
            } else if scopes.last().unwrap().is_accepting_lines() {
                if let Some(cap) = SHADER_IMPORT_PROCESSOR
                    .import_name_regex
                    .captures(line)
                {
                    let import = ShaderImport::Name(cap.get(1).unwrap().as_str().to_string());
                    self.apply_import(
                        &import,
                        shader_defs_unique,
                        write_to,
                    )?;
                } else if let Some(cap) = SHADER_IMPORT_PROCESSOR
                    .import_full_path_regex
                    .captures(line)
                {
                    let import = ShaderImport::FullPath(cap.get(1).unwrap().as_str().to_string());
                    self.apply_import(
                        &import,
                        shader_defs_unique,
                        write_to,
                    )?;
                } else if SHADER_IMPORT_PROCESSOR
                    .define_import_path_regex
                    .is_match(line)
                {
                } else if let Some(cap) = self.define_regex.captures(line) {
                    let def = cap.get(1).unwrap();
                    let name = def.as_str().to_string();

                    if let Some(val) = cap.get(2) {
                        if let Ok(val) = val.as_str().parse::<u32>() {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::UInt(name, val));
                        } else if let Ok(val) = val.as_str().parse::<i32>() {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::Int(name, val));
                        } else if let Ok(val) = val.as_str().parse::<bool>() {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::Bool(name, val));
                        } else if let Ok(val) = val.as_str().parse::<f32>() {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::Float(name, val));
                        } else {
                            shader_defs_unique.insert(name.clone(), ShaderDefVal::Any(name, val.as_str().to_string()));
                        }
                    } else {
                        shader_defs_unique.insert(name.clone(), ShaderDefVal::Bool(name, true));
                    }
                } else {
                    let mut line_with_defs = line.to_string();
                    for capture in self.def_regex.captures_iter(line) {
                        let def = capture.get(1).unwrap();
                        if let Some(def) = shader_defs_unique.get(def.as_str()).and_then(|def| def.value_as_string()) {
                            line_with_defs = self
                                .def_regex
                                .replace(&line_with_defs, def)
                                .to_string();
                        }
                    }
                    for capture in self.def_regex_delimited.captures_iter(line) {
                        let def = capture.get(1).unwrap();
                        if let Some(def) = shader_defs_unique.get(def.as_str()).and_then(|def| def.value_as_string()) {
                            line_with_defs = self
                                .def_regex_delimited
                                .replace(&line_with_defs, def)
                                .to_string();
                        }
                    }
                    write_to.push_str(&line_with_defs);
                    write_to.push('\n');
                }
            }
        }

        if scopes.len() != 1 {
            return Err(ProcessShaderError::NotEnoughEndIfs)
        }
        
        if for_scopes.len() != 1 {
            return Err(ProcessShaderError::NotEnoughEndFors)
        }

        Ok(for_scopes.remove(0).string)
    }

    fn apply_import(
        &self,
        import: &ShaderImport,
        shader_defs_unique: &mut HashMap<String, ShaderDefVal>,
        final_string: &mut String,
    ) -> Result<(), ProcessShaderError> {
        let imported_shader = self.all_shaders
            .get(import)
            .ok_or_else(|| ProcessShaderError::UnresolvedImport(import.clone()))?;

        let imported_processed =
            self.process_inner(imported_shader, shader_defs_unique);
        
        let imported_processed = match imported_processed{
            Ok(val) => val,
            Err(err) => return Err(err.clone()),
        };


        final_string.push_str(&imported_processed);
        
        // match &shader.source {
        //     Source::Wgsl(_) => {
        //         if let ProcessedShader::Wgsl(import_source) = &imported_processed {
        //             final_string.push_str(import_source);
        //         } else {
        //             return Err(ProcessShaderError::MismatchedImportFormat(import.clone()));
        //         }
        //     }
        //     Source::Glsl(_, _) => {
        //         if let ProcessedShader::Glsl(import_source, _) = &imported_processed {
        //             final_string.push_str(import_source);
        //         } else {
        //             return Err(ProcessShaderError::MismatchedImportFormat(import.clone()));
        //         }
        //     }
        //     Source::SpirV(_) => {
        //         return Err(ProcessShaderError::ShaderFormatDoesNotSupportImports);
        //     }
        // }

        Ok(())
    }
}

struct ForScope{
    ident: String,
    string: String,
}

struct IfScope {
    // Is the current scope one in which we should accept new lines into the output?
    accepting_lines: bool,

    // Has this scope ever accepted lines?
    // Needs to be tracked for #else ifdef chains.
    has_accepted_lines: bool,
}

impl IfScope {
    fn new(should_lines_be_accepted: bool) -> Self {
        Self {
            accepting_lines: should_lines_be_accepted,
            has_accepted_lines: should_lines_be_accepted,
        }
    }

    fn is_accepting_lines(&self) -> bool {
        self.accepting_lines
    }

    fn stop_accepting_lines(&mut self) {
        self.accepting_lines = false;
    }

    fn start_accepting_lines_if_appropriate(&mut self) {
        if !self.has_accepted_lines {
            self.has_accepted_lines = true;
            self.accepting_lines = true;
        } else {
            self.accepting_lines = false;
        }
    }
}

fn _add_directory<P: AsRef<Path>>(
    shaders: &mut HashMap<ShaderImport, Shader>,
    path: P,
    full_path: bool,
) -> Result<(), std::io::Error>{
    for file in fs::read_dir(path)?{
        let file = file?;
        if !file.metadata()?.is_file(){
            continue
        }
        let file = file.path();
        let Some(ext) = file.extension()
            .and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase()) else { continue };
        if ext != "wgsl"{
            continue
        }

        let Ok(source_string) = std::fs::read_to_string(&file) else { continue };
        let file = if !full_path{
            file.file_stem().unwrap().to_os_string()
        } else {
            file.into_os_string()
        };
        let Ok(file) = file.into_string() else { continue };
        let shaderimport = ShaderImport::Name(file);
        let shader = Shader::from_wgsl(source_string);
        shaders.insert(shaderimport, shader);
    }
    Ok(())
}

impl From<HashMap<ShaderImport, Shader>> for ShaderProcessor{
    fn from(value: HashMap<ShaderImport, Shader>) -> Self {
        Self {
            all_shaders: value,
            ..Default::default()
        }
    }
}


#[derive(Debug, Clone)]
pub struct ShaderSpecs{
    pub workgroup_size: WorkgroupSize,
    pub dispatcher: Option<Dispatcher>,
    pub push_constants: Option<u32>,
    pub shader_defs: Vec<ShaderDefVal>,
    pub entry_point: Option<String>,
    
    pub shader_label: Option<String>,
    pub bindgroup_layout_label: Option<String>,
    pub pipelinelayout_label: Option<String>,
    pub pipeline_label: Option<String>,
}

impl ShaderSpecs{
    pub fn new(workgroup_size: impl Into<WorkgroupSize>) -> Self{
        let workgroup_size = workgroup_size.into();
        let shader_defs = vec![
            ShaderDefVal::UInt(workgroup_size.x_name.clone(), workgroup_size.x),
            ShaderDefVal::UInt(workgroup_size.y_name.clone(), workgroup_size.y),
            ShaderDefVal::UInt(workgroup_size.z_name.clone(), workgroup_size.z),
        ];
        Self{
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

    pub fn workgroupsize(mut self, val: WorkgroupSize) -> Self{
        self.workgroup_size = val;
        self
    }
    
    pub fn dispatcher(mut self, val: Dispatcher) -> Self{
        self.dispatcher = Some(val);
        self
    }

    pub fn direct_dispatcher(mut self, dims: &[u32; 3]) -> Self{
        self.dispatcher = Some(Dispatcher::new_direct(dims, &self.workgroup_size));
        self
    }
    
    pub fn push_constants(mut self, val: u32) -> Self{
        self.push_constants = Some(val);
        self
    }

    pub fn extend_defs(mut self, vals: &[ShaderDefVal]) -> Self{
        self.shader_defs.extend_from_slice(vals);
        self
    }

    pub fn shader_label(mut self, val: &str) -> Self{
        self.shader_label = Some(val.to_string());
        self
    }

    pub fn bindgroup_layout_label(mut self, val: &str) -> Self{
        self.bindgroup_layout_label = Some(val.to_string());
        self
    }

    pub fn pipelinelayout_label(mut self, val: &str) -> Self{
        self.pipelinelayout_label = Some(val.to_string());
        self
    }

    pub fn pipeline_label(mut self, val: &str) -> Self{
        self.pipeline_label = Some(val.to_string());
        self
    }

    pub fn entry_point(mut self, val: &str) -> Self{
        self.entry_point = Some(val.to_string());
        self
    }

    pub fn labels(self, val: &str) -> Self{
        self.shader_label(val)
            .bindgroup_layout_label(val)
            .pipelinelayout_label(val)
            .pipeline_label(val)
    }

}

#[derive(Debug)]
pub struct NonBoundPipeline{
    pub label: Option<String>,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub dispatcher: Option<Dispatcher>,
}


pub mod tests{
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_read_dir(){
        // let mut map = Default::default();
        // add_directory(&mut map, ".");
        // dbg!(map);
        
        let mut map = Default::default();
        dbg!(_add_directory(&mut map, r"src\test_shaders", false)).unwrap();
        dbg!(&map);
        let _processor = ShaderProcessor::default();

        for (_, _shader) in map.iter(){
            // let idk = processor.process(shader, &[]);
            // dbg!(idk.unwrap());
        }
        // processor.process(shader, , )
    }
}
