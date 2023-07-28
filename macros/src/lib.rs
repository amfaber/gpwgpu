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
            #vis static #processor: #prefix::Lazy<#pp_ty> = #prefix::Lazy::new(|| #pp_ty(
                std::collections::HashMap::<
                std::borrow::Cow<str>, #prefix::shaderpreprocessor::ParsedShader,
                >::from(#expr_array)
            ));
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

// #[proc_macro_attribute]
// pub fn logical_buffers(_attr: TokenStream, input: TokenStream) -> TokenStream {
//     let mut out = quote!(#[derive(Hash, Eq, PartialEq, Clone, Debug, Copy)]);
//     let mut e: ItemEnum = syn::parse(input.clone()).unwrap();
//     // let out = quote!();
//     let bufimpl = impl_buffers(&mut e);
//     out.extend(e.to_token_stream());
//     out.extend(bufimpl);
//     out.into()
// }

// fn impl_buffers(ast: &mut ItemEnum) -> proc_macro2::TokenStream {
//     let name = &ast.ident;
    
//     let mut sizes = quote!();
    
//     let mut usages = quote!();
    
//     let mut reqs = quote!();

    
//     for variant in ast.variants.iter_mut(){
//         let variant_name = &variant.ident;
//         let BufAttribute { size, usage, req } = parse_attributes(&mut variant.attrs);
        
//         sizes.extend(quote!(#name::#variant_name => #size,));
//         usages.extend(quote!(#name::#variant_name => #usage,));
//         reqs.extend(quote!(#name::#variant_name => #req,));
//     }

//     quote! {
//         impl gpwgpu::automatic_buffers::LogicalBuffers for #name {
//             fn size(&self) -> wgpu::BufferAddress{
//                 match self {
//                     #sizes
//                 }
//             }
            
//             fn usage(&self) -> wgpu::BufferUsages{
//                 match self {
//                     #usages
//                 }
//             }
            
//             fn memory_req(&self) -> gpwgpu::automatic_buffers::MemoryReq {
//                 match self {
//                     #reqs
//                 }
//             }
//         }
//     }
// }

// fn parse_attributes(attrs: &mut Vec<Attribute>) -> BufAttribute {

//     let msg = "Every buffer must have the #[buf(size = {int}, usages = {wgpu::BufferUsages}, req = {gpwgpu::automatic_buffers::MemoryReq})] attribute";

//     let Some(idx) = attrs.iter().position(|x| x.path().is_ident("buf")) else {panic!("{}", msg)};
//     let Attribute {
//         meta : Meta::List(MetaList {
//             tokens,
//             ..
//         }),
//         .. 
//     } = attrs.remove(idx) else {panic!("{}", msg)};

    
//     syn::parse2(tokens.clone()).unwrap()
// }

// struct BufAttribute {
//     size: u64,
//     usage: syn::Expr,
//     req: syn::Expr,
// }

// impl syn::parse::Parse for BufAttribute {
//     fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
//         let mut size: Option<wgpu::BufferAddress> = None;
//         let mut usage: Option<Expr> = None;
//         let mut req: Option<Expr> = None;

//         while !input.is_empty() {
//             let key: syn::Ident = input.parse()?;
//             input.parse::<syn::Token![=]>()?;
//             if key == "size" {
//                 let lit: syn::LitInt = input.parse()?;
//                 size = Some(lit.base10_parse()?);
//             } else if key == "usages" {
//                 let expr: syn::Expr = input.parse()?;
//                 usage = Some(expr);
//             } else if key == "req" {
//                 let expr: syn::Expr = input.parse()?;
//                 req = Some(expr);
//             } else {
//                 return Err(input.error("Expected 'size', 'usages', or 'req'"));
//             }

//             // Allow comma-separated attributes
//             let _ = input.parse::<syn::Token![,]>();
//         }
//         Ok(BufAttribute {
//             size: size.ok_or(input.error("size not supplied"))?,
//             usage: usage.ok_or(input.error("usages not supplied"))?,
//             req: req.ok_or(input.error("req not supplied"))?,
//         })
//     }
// }
