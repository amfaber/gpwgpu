#![allow(unused_imports)]
use std::{
    collections::HashMap,
    fs::{self, DirEntry},
    path::PathBuf,
};

use my_core::{
    shaderpreprocessor::{Shader, ShaderImport, ShaderProcessor},
    *,
};
use proc_macro::TokenStream;
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
        let Some((name, file)) = ShaderProcessor::validate_file(file, full_path) else { continue };
        let mut abs = current_dir.clone();
        abs.push(file);
        let Ok(abs) = abs.into_os_string().into_string() else { continue };
        let expr: Expr = parse_quote!((#prefix::ShaderImport::Name(#name.to_string()),
            #prefix::Shader::from_wgsl(include_str!(#abs))));
        out.elems.push(expr)
    }

    Ok(out)
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

#[proc_macro]
pub fn add_directory(inp: TokenStream) -> TokenStream {
    internal_add_directory(inp, quote!(gpwgpu::shaderpreprocessor))
}

#[proc_macro]
pub fn add_directory_crate(inp: TokenStream) -> TokenStream {
    internal_add_directory(inp, quote!(crate::shaderpreprocessor))
}
