use std::{error::Error, fs, path::Path, process::ExitCode};

use syn::{Item, ItemStruct, Meta, Token, Type, punctuated::Punctuated, spanned::Spanned};

#[derive(Default)]
struct ConfigLints {
    violations: Vec<(usize, usize, String)>,
}

impl ConfigLints {
    fn check_items(&mut self, items: &[Item]) {
        let mut configs = std::collections::HashSet::new();
        for item in items {
            let Item::Impl(item) = item else { continue };
            let Some((_, trait_path, _)) = &item.trait_ else {
                continue;
            };
            if !is_config_path(trait_path, "TomlConfig") {
                continue;
            }
            if let Type::Path(target) = item.self_ty.as_ref()
                && let Some(name) = target.path.get_ident()
                && items
                    .iter()
                    .any(|item| matches!(item, Item::Struct(item) if item.ident == *name))
            {
                configs.insert(name);
                continue;
            }
            let location = trait_path.segments.last().unwrap().ident.span().start();
            self.violations.push((
                location.line,
                location.column + 1,
                "toml_config_impl: put the TomlConfig implementation next to its struct".into(),
            ));
        }
        for item in items {
            match item {
                Item::Struct(item)
                    if configs.contains(&item.ident) || is_annotated_toml_config(item) =>
                {
                    self.check_fields(item);
                }
                Item::Mod(item) => {
                    if let Some((_, items)) = &item.content {
                        self.check_items(items);
                    }
                }
                _ => {}
            }
        }
    }

    fn check_fields(&mut self, item: &ItemStruct) {
        for (index, field) in item.fields.iter().enumerate() {
            let name = field
                .ident
                .as_ref()
                .map_or_else(|| index.to_string(), ToString::to_string);
            if is_option_type(&field.ty) {
                let location = field.ty.span().start();
                self.violations.push((
                    location.line,
                    location.column + 1,
                    format!(
                        "toml_optional_field: {}.{name} must not use Option\n  = reason: a tagged enum makes the available modes and the settings for each mode explicit.\n  = help: use a tagged enum and select a variant in the checked-in TOML",
                        item.ident
                    ),
                ));
            }
            for attr in &field.attrs {
                if !attr.path().is_ident("serde") {
                    continue;
                }
                let options = attr
                    .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                    .expect("Clippy validates Serde attributes before this lint runs");
                for option in options
                    .iter()
                    .filter(|option| option.path().is_ident("default"))
                {
                    let location = option.path().segments[0].ident.span().start();
                    self.violations.push((
                        location.line,
                        location.column + 1,
                        format!(
                            "serde_field_default: {}.{name} must not declare a Serde default\n  = reason: config deserializers target the TOML files checked into this repository. Older config formats require no backward compatibility.\n  = help: remove the Serde default and set the default value in the checked-in TOML file",
                            item.ident
                        ),
                    ));
                }
            }
        }
    }
}

fn is_option_type(ty: &Type) -> bool {
    match ty {
        Type::Path(ty) if ty.qself.is_none() => {
            let segments = &ty.path.segments;
            segments.last().is_some_and(|last| last.ident == "Option")
                && (segments.len() == 1
                    || (segments.len() == 3
                        && (segments[0].ident == "std" || segments[0].ident == "core")
                        && segments[1].ident == "option"))
        }
        Type::Paren(ty) => is_option_type(&ty.elem),
        Type::Group(ty) => is_option_type(&ty.elem),
        _ => false,
    }
}

fn is_config_path(path: &syn::Path, name: &str) -> bool {
    path.is_ident(name)
        || (path.segments.len() == 2
            && path.segments[0].ident == "chess_config"
            && path.segments[1].ident == name)
}

fn is_annotated_toml_config(item: &ItemStruct) -> bool {
    item.attrs.iter().any(|attr| {
        is_config_path(attr.path(), "toml_config")
            || (attr.path().is_ident("derive")
                && attr
                    .parse_args_with(Punctuated::<syn::Path, Token![,]>::parse_terminated)
                    .is_ok_and(|derives| {
                        derives
                            .iter()
                            .any(|path| is_config_path(path, "TomlConfig"))
                    }))
    })
}

fn check_tree(directory: &Path) -> Result<bool, Box<dyn Error>> {
    let mut paths = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort_unstable();

    let mut clean = true;
    for path in paths {
        if path.is_dir() {
            clean &= check_tree(&path)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = fs::read_to_string(&path)?;
            let file =
                syn::parse_file(&source).map_err(|error| format!("{}: {error}", path.display()))?;
            let mut lint = ConfigLints::default();
            lint.check_items(&file.items);
            for (line, column, message) in lint.violations {
                eprintln!("{}:{line}:{column}: {message}", path.display());
                clean = false;
            }
        }
    }
    Ok(clean)
}

fn main() -> ExitCode {
    match check_tree(Path::new("crates")) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_defaults_only_on_toml_config_structs() {
        let file = syn::parse_file(
            r#"
            #[derive(serde::Deserialize)]
            struct Output { #[serde(default)] x: u32 }

            mod nested {
                #[chess_config::toml_config]
                #[serde(default)]
                struct Settings {
                    #[serde(rename = "x", default)]
                    named: u32,
                    optional: Option<u32>,
                }

                #[derive(Deserialize)]
                struct Tuple(#[serde(default = "fallback")] u32);
                impl TomlConfig for Tuple {}
            }
        "#,
        )
        .unwrap();
        let mut lint = ConfigLints::default();
        lint.check_items(&file.items);
        let hits: Vec<_> = lint
            .violations
            .iter()
            .map(|(line, column, message)| (*line, *column, message.lines().next().unwrap()))
            .collect();
        assert_eq!(
            hits,
            [
                (
                    9,
                    43,
                    "serde_field_default: Settings.named must not declare a Serde default"
                ),
                (
                    11,
                    31,
                    "toml_optional_field: Settings.optional must not use Option"
                ),
                (
                    15,
                    38,
                    "serde_field_default: Tuple.0 must not declare a Serde default"
                ),
            ]
        );
    }

    #[test]
    fn detects_option_fields_only_on_toml_config_structs() {
        let file = syn::parse_file(
            r#"
            struct Runtime { optional: Option<u32> }

            #[derive(Deserialize, chess_config::TomlConfig)]
            struct Settings {
                std_option: std::option::Option<u32>,
                core_option: ::core::option::Option<u32>,
                collection: Vec<Option<u32>>,
                custom: other::Option<u32>,
            }

            struct Tuple(Option<u32>);
            impl TomlConfig for Tuple {}
            "#,
        )
        .unwrap();
        let mut lint = ConfigLints::default();
        lint.check_items(&file.items);
        let hits: Vec<_> = lint
            .violations
            .iter()
            .map(|(_, _, message)| message.lines().next().unwrap())
            .collect();
        assert_eq!(
            hits,
            [
                "toml_optional_field: Settings.std_option must not use Option",
                "toml_optional_field: Settings.core_option must not use Option",
                "toml_optional_field: Tuple.0 must not use Option",
            ]
        );
    }
}
