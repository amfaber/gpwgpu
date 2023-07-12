#![cfg_attr(feature = "nightly", feature(track_path))]
#![allow(unused_imports)]
use std::{
    collections::HashMap,
    fs::{self, DirEntry},
    path::PathBuf,
};

use my_core::{
    shaderpreprocessor::{ShaderProcessor, validate_wgsl_file, parse_shader},
    *, parser::parse_tokens,
};
use proc_macro::TokenStream;
#[cfg(feature = "nightly")]
use proc_macro::tracked_path;
use proc_macro2::extra::DelimSpan;
use quote::quote;
use rust_format::{Formatter, RustFmt};
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::Bracket,
    *,
};

macro_rules! unpack {
    ($inp:expr) => {
        match $inp {
            Ok(val) => val,
            Err(err) => return err.into_compile_error().into(),
        }
    };
}

struct ProcessorAndDir {
    processor: Ident,
    dir: Dir,
}

type Dir = LitStr;

impl Parse for ProcessorAndDir {
    fn parse(input: ParseStream) -> Result<Self> {
        let processor = input.parse()?;
        let _: Token![,] = input.parse()?;
        let dir = input.parse()?;

        Ok(Self { processor, dir })
    }
}

fn read_directory(
    dir: &Dir,
    full_path: bool,
    prefix: &proc_macro2::TokenStream,
) -> Result<ExprArray> {
    let path = dir.value();
    let span = dir.span();
    let Ok(read_dir) = fs::read_dir(path) else {
        return Err(Error::new(span, "Could not read directory"))
    };
    let mut out: ExprArray = parse_quote!([]);
    let Ok(current_dir) = std::env::current_dir() else{
        return Err(Error::new(span, "Could not read current directory"))
    };

    for file in read_dir {
        let Some((name, file)) = validate_wgsl_file(file, full_path) else { continue };
        let mut abs = current_dir.clone();
        abs.push(file);
        let Ok(abs) = abs.into_os_string().into_string() else { continue };
        let expr: Expr = parse_quote!((#prefix::ShaderImport::Name(#name.to_string()),
            #prefix::Shader::from_wgsl(include_str!(#abs))));
        out.elems.push(expr)
    }

    Ok(out)
}

fn read_directory_parser(
    dir: &Dir,
    full_path: bool,
    prefix: &proc_macro2::TokenStream,
) -> Result<ExprArray> {
    let path = dir.value();
    let span = dir.span();
    let Ok(read_dir) = fs::read_dir(path) else {
        return Err(Error::new(span, "Could not read directory"))
    };
    let mut out: ExprArray = parse_quote!([]);
    let Ok(current_dir) = std::env::current_dir() else{
        return Err(Error::new(span, "Could not read current directory"))
    };

    for file in read_dir {
        let Some((name, file)) = validate_wgsl_file(file, full_path) else { continue };
        
        let mut abs = current_dir.clone();
        abs.push(file);
        let Ok(abs) = abs.into_os_string().into_string() else { continue };

        let Ok(shader) = fs::read_to_string(&abs) else { continue };
        #[cfg(feature = "nightly")]
        tracked_path::path(&abs);

        let tokens = match parse_shader(&shader){
            Ok(tokens) => tokens,
            Err(e) => panic!("Failed at parsing shader {}\nerror: {}", abs, e)
        };
        let ser = bincode::serialize(&tokens.0).unwrap();
        let mut ser_array: ExprArray = parse_quote!([]);
        for datum in ser{
            ser_array.elems.push(parse_quote!(#datum));
        }
        
        let expr: Expr = parse_quote!((#name,
            #prefix::shaderpreprocessor::ParsedShader(
                bincode::deserialize::<Vec<#prefix::parser::Token>>(&#ser_array).unwrap()
            )
        ));
        out.elems.push(expr)
    }

    Ok(out)
}

fn internal_parse_shaders(input: TokenStream, prefix: proc_macro2::TokenStream) -> TokenStream{
    let dir = parse_macro_input!(input as Dir);
    let expr_array = unpack!(read_directory_parser(&dir, false, &prefix));
    let hashmap = quote!(
        #prefix::shaderpreprocessor::ShaderProcessor(
            std::collections::HashMap::<
            &str, #prefix::shaderpreprocessor::ParsedShader,
            >::from(#expr_array)
        )
    );
    hashmap.into()
}

#[proc_macro]
pub fn parse_shaders_crate(input: TokenStream) -> TokenStream{
    internal_parse_shaders(input, quote!(crate))
}

#[proc_macro]
pub fn parse_shaders(input: TokenStream) -> TokenStream{
    internal_parse_shaders(input, quote!(gpwgpu))
}

fn internal_add_directory(inp: TokenStream, prefix: proc_macro2::TokenStream) -> TokenStream {
    let (processor, directory) = match syn::parse::<ProcessorAndDir>(inp.clone()) {
        Ok(ProcessorAndDir { processor, dir }) => (Some(processor), dir),
        Err(_) => {
            let processor = None;
            let dir = parse_macro_input!(inp as Dir);
            (processor, dir)
        }
    };

    let out = unpack!(read_directory(&directory, false, &prefix));
    let hashmap = quote!(std::collections::HashMap::<
        #prefix::ShaderImport,
        #prefix::Shader,
    >::from(#out));

    let out = match processor {
        Some(ident) => quote!(#ident.all_shaders.extend(#hashmap)),
        None => quote!(#prefix::ShaderProcessor::from(#hashmap)),
    };

    // let formatter = RustFmt::default();
    // let prettied = formatter.format_str(out.to_string()).unwrap_or(out.to_string());
    // std::fs::write("macros/expansions/expanded.rs", prettied).unwrap();

    out.into()
}



// #[proc_macro]
// pub fn add_directory()

#[proc_macro]
pub fn add_directory(inp: TokenStream) -> TokenStream {
    internal_add_directory(inp, quote!(gpwgpu::shaderpreprocessor))
}

#[proc_macro]
pub fn add_directory_crate(inp: TokenStream) -> TokenStream {
    internal_add_directory(inp, quote!(crate::shaderpreprocessor))
}
