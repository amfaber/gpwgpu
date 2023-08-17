#![cfg_attr(feature = "nightly", feature(track_path))]
#![allow(unused_imports)]
use std::{
    collections::HashMap,
    fs::{self, DirEntry},
    path::PathBuf,
};

use my_core::{
    parser::parse_tokens,
    shaderpreprocessor::{parse_shader, validate_wgsl_file, ShaderProcessor},
    *,
};
#[cfg(feature = "nightly")]
use proc_macro::tracked_path;
use proc_macro::TokenStream;
use proc_macro2::extra::DelimSpan;
use quote::{quote, ToTokens};
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
    vis: Option<Visibility>,
    processor: Ident,
    dir: Dir,
}

impl Parse for ProcessorAndDir {
    fn parse(input: ParseStream) -> Result<Self> {
        let vis: Option<Visibility> = input.parse().ok();
        let processor = input.parse()?;
        let _: Token![,] = input.parse()?;
        let dir = input.parse()?;

        Ok(Self {
            vis,
            processor,
            dir,
        })
    }
}

type Dir = LitStr;

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

        let tokens = match parse_shader(&shader) {
            Ok(tokens) => tokens,
            Err(e) => panic!("Failed at parsing shader {}\nerror: {}", abs, e),
        };
        let ser = bincode::serialize(&tokens.0).unwrap();
        let mut ser_array: ExprArray = parse_quote!([]);
        for datum in ser {
            ser_array.elems.push(parse_quote!(#datum));
        }

        let expr: Expr = parse_quote!((#name.into(),
            #prefix::shaderpreprocessor::ParsedShader(
                #prefix::bincode::deserialize::<Vec<#prefix::parser::Token>>(&#ser_array).unwrap()
            )
        ));
        out.elems.push(expr)
    }

    Ok(out)
}

fn internal_parse_shaders(
    input: TokenStream,
    prefix: proc_macro2::TokenStream,
    is_static: bool,
) -> TokenStream {
    let ProcessorAndDir {
        vis,
        processor,
        dir,
    } = parse_macro_input!(input as ProcessorAndDir);
    let pp_ty = quote!(#prefix::shaderpreprocessor::ShaderProcessor);
    if is_static {
        let expr_array = unpack!(read_directory_parser(&dir, false, &prefix));
        let hashmap = quote!(
            #vis static #processor: #prefix::Lazy<#pp_ty> = #prefix::Lazy::new(|| #pp_ty::from_parsed_shader_hashmap(
                std::collections::HashMap::<
                std::borrow::Cow<str>, #prefix::shaderpreprocessor::ParsedShader,
                >::from(#expr_array)
            ).unwrap());
        );
        hashmap.into()
    } else {
        quote!(
            #vis static #processor: #prefix::Lazy<#pp_ty> = #prefix::Lazy::new(|| #pp_ty::load_dir_dyn(
                #dir
            ).unwrap());
        ).into()
    }
}

#[proc_macro]
pub fn parse_shaders_crate(input: TokenStream) -> TokenStream {
    internal_parse_shaders(input, quote!(crate), true)
}

#[proc_macro]
pub fn parse_shaders(input: TokenStream) -> TokenStream {
    internal_parse_shaders(input, quote!(gpwgpu), true)
}

#[proc_macro]
pub fn parse_shaders_dyn(input: TokenStream) -> TokenStream {
    internal_parse_shaders(input, quote!(gpwgpu), false)
}

