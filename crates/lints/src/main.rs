use std::{error::Error, fs, path::Path, process::ExitCode};

use syn::{ItemStruct, Meta, Token, punctuated::Punctuated, visit::Visit};

#[derive(Default)]
struct SerdeFieldDefaults {
    violations: Vec<(usize, usize, String)>,
}

impl<'ast> Visit<'ast> for SerdeFieldDefaults {
    fn visit_item_struct(&mut self, item: &'ast ItemStruct) {
        let deserializes = item.attrs.iter().any(|attr| {
            attr.path().is_ident("derive")
                && attr
                    .parse_args_with(Punctuated::<syn::Path, Token![,]>::parse_terminated)
                    .is_ok_and(|paths| {
                        paths.iter().any(|path| {
                            path.is_ident("Deserialize")
                                || (path.segments.len() == 2
                                    && path.segments[0].ident == "serde"
                                    && path.segments[1].ident == "Deserialize")
                        })
                    })
        });
        if !deserializes {
            return;
        }
        for (index, field) in item.fields.iter().enumerate() {
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
                    let name = field
                        .ident
                        .as_ref()
                        .map_or_else(|| index.to_string(), ToString::to_string);
                    self.violations.push((
                        location.line,
                        location.column + 1,
                        format!("{}.{}", item.ident, name),
                    ));
                }
            }
        }
    }
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
            let mut lint = SerdeFieldDefaults::default();
            lint.visit_file(&file);
            for (line, column, field) in lint.violations {
                eprintln!(
                    "{}:{line}:{column}: serde_field_default: {field} must not declare a Serde default\n  = reason: config deserializers target the TOML files checked into this repository. Older config formats require no backward compatibility.\n  = help: remove the Serde default and set the default value in the checked-in TOML file",
                    path.display()
                );
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
    fn detects_explicit_field_defaults_on_deserialized_structs() {
        let file = syn::parse_file(
            r#"
            #[derive(Serialize)]
            struct Output { #[serde(default)] x: u32 }

            mod nested {
                #[derive(serde::Deserialize)]
                #[serde(default)]
                struct Settings {
                    #[serde(rename = "x", default)]
                    named: u32,
                    optional: Option<u32>,
                }

                #[derive(Deserialize)]
                struct Tuple(#[serde(default = "fallback")] u32);
            }
        "#,
        )
        .unwrap();
        let mut lint = SerdeFieldDefaults::default();
        lint.visit_file(&file);
        assert_eq!(
            lint.violations,
            [(9, 43, "Settings.named".into()), (15, 38, "Tuple.0".into())]
        );
    }
}
