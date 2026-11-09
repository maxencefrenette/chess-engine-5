use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, ItemStruct, parse::Nothing, parse_macro_input};

#[proc_macro_derive(TomlConfig)]
pub fn derive_toml_config(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    quote! {
        impl #impl_generics ::chess_config::TomlConfig for #name #type_generics #where_clause {}
    }
    .into()
}

#[proc_macro_attribute]
pub fn toml_config(args: TokenStream, item: TokenStream) -> TokenStream {
    parse_macro_input!(args as Nothing);
    let item = parse_macro_input!(item as ItemStruct);
    quote! {
        #[derive(::serde::Deserialize, ::chess_config::TomlConfig)]
        #[serde(deny_unknown_fields)]
        #item
    }
    .into()
}
