use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{DeriveInput, parse::Nothing, parse_macro_input};

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

/// Derive strict TOML deserialization for a struct or enum.
/// On f64 fields, `#[range(min = 0.0, max_exclusive = 1.0)]` validates
/// finite values during deserialization. Either endpoint may be omitted;
/// `min_exclusive` and `max` select the other endpoint conventions.
#[proc_macro_attribute]
pub fn toml_config(args: TokenStream, item: TokenStream) -> TokenStream {
    parse_macro_input!(args as Nothing);
    let mut item = parse_macro_input!(item as DeriveInput);
    match expand_config(&mut item) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand_config(item: &mut DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &item.ident;
    let mut helpers = Vec::new();
    let groups: Vec<_> = match &mut item.data {
        syn::Data::Struct(data) => vec![(String::new(), &mut data.fields)],
        syn::Data::Enum(data) => data
            .variants
            .iter_mut()
            .map(|variant| (variant.ident.to_string(), &mut variant.fields))
            .collect(),
        syn::Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                name,
                "TOML configs cannot be unions",
            ));
        }
    };
    for (variant, fields) in groups {
        for (index, field) in fields.iter_mut().enumerate() {
            let mut checks = Vec::new();
            let mut lower = false;
            let mut upper = false;
            let mut ranged = false;
            for attr in &field.attrs {
                if !attr.path().is_ident("range") {
                    continue;
                }
                ranged = true;
                attr.parse_nested_meta(|meta| {
                    let (seen, op) = if meta.path.is_ident("min") {
                        (&mut lower, quote!(>=))
                    } else if meta.path.is_ident("min_exclusive") {
                        (&mut lower, quote!(>))
                    } else if meta.path.is_ident("max") {
                        (&mut upper, quote!(<=))
                    } else if meta.path.is_ident("max_exclusive") {
                        (&mut upper, quote!(<))
                    } else {
                        return Err(
                            meta.error("expected min, min_exclusive, max, or max_exclusive")
                        );
                    };
                    if *seen {
                        return Err(meta.error("duplicate range endpoint"));
                    }
                    *seen = true;
                    let bound: syn::Expr = meta.value()?.parse()?;
                    checks.push(quote!(value #op (#bound)));
                    Ok(())
                })?;
            }
            if !ranged {
                continue;
            }
            if !matches!(&field.ty, syn::Type::Path(ty) if ty.path.is_ident("f64")) {
                return Err(syn::Error::new_spanned(
                    &field.ty,
                    "range requires an f64 field",
                ));
            }
            if checks.is_empty() {
                return Err(syn::Error::new_spanned(
                    field,
                    "range requires at least one endpoint",
                ));
            }
            field.attrs.retain(|attr| !attr.path().is_ident("range"));
            let label = field
                .ident
                .as_ref()
                .map_or(index.to_string(), ToString::to_string);
            let helper = format_ident!("__range_{}_{}_{}", name, variant, label);
            let path = helper.to_string();
            field
                .attrs
                .push(syn::parse_quote!(#[serde(deserialize_with = #path)]));
            helpers.push(quote! {
                #[allow(non_snake_case, clippy::manual_range_contains)]
                fn #helper<'de, D: ::serde::Deserializer<'de>>(deserializer: D) -> ::std::result::Result<f64, D::Error> {
                    let value = <f64 as ::serde::Deserialize>::deserialize(deserializer)?;
                    if value.is_finite() #(&& #checks)* {
                        Ok(value)
                    } else {
                        Err(<D::Error as ::serde::de::Error>::custom(format!(
                            "{} must be finite and satisfy {}; got {}",
                            #label, stringify!(#(#checks)&&*), value
                        )))
                    }
                }
            });
        }
    }
    Ok(quote! {
        #[derive(::serde::Deserialize, ::chess_config::TomlConfig)]
        #[serde(deny_unknown_fields)]
        #item
        #(#helpers)*
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_range_annotations() {
        for source in [
            "struct C { #[range(min = 0.0, min_exclusive = 1.0)] x: f64 }",
            "struct C { #[range(max = 1.0, max = 2.0)] x: f64 }",
            "struct C { #[range(minimum = 0.0)] x: f64 }",
            "struct C { #[range()] x: f64 }",
            "struct C { #[range(min = 0)] x: usize }",
        ] {
            let mut item = syn::parse_str(source).unwrap();
            assert!(expand_config(&mut item).is_err(), "{source}");
        }
    }
}
