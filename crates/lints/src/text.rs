use std::{collections::BTreeSet, error::Error, fs, io, path::Path, process::Command};

use std::os::unix::ffi::OsStrExt;

use regex::Regex;

mod rules;

struct Rule {
    name: &'static str,
    pattern: &'static str,
    help: &'static str,
    reason: &'static str,
}

pub fn check_repository() -> Result<bool, Box<dyn Error>> {
    let output = Command::new("git")
        .args([
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(String::from_utf8_lossy(&output.stderr)).into());
    }
    let paths: BTreeSet<_> = output
        .stdout
        .split(|&byte| byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| Path::new(std::ffi::OsStr::from_bytes(path)))
        .collect();
    let rules = rules::RULES
        .iter()
        .map(|rule| Ok((rule, Regex::new(rule.pattern)?)))
        .collect::<Result<Vec<_>, regex::Error>>()?;
    let mut clean = true;
    for path in paths {
        // Rule definitions and their fixtures must be able to contain banned wording.
        if path == Path::new("crates/lints/src/text/rules.rs") {
            continue;
        }
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_file() {
            continue;
        }
        let bytes = fs::read(path)?;
        let Some(source) = text_source(&bytes) else {
            continue;
        };
        for (rule, regex) in &rules {
            for hit in regex.find_iter(source) {
                let (line, column) = location(source, hit.start());
                eprintln!(
                    "{}:{line}:{column}: {}: {}\n  = reason: {}",
                    path.display(),
                    rule.name,
                    rule.help,
                    rule.reason
                );
                clean = false;
            }
        }
    }
    Ok(clean)
}

fn text_source(bytes: &[u8]) -> Option<&str> {
    if bytes.contains(&0) {
        return None;
    }
    std::str::from_utf8(bytes).ok()
}

fn location(source: &str, offset: usize) -> (usize, usize) {
    let prefix = &source[..offset];
    let line = prefix.bytes().filter(|&byte| byte == b'\n').count() + 1;
    let column = prefix.rsplit('\n').next().unwrap().chars().count() + 1;
    (line, column)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_binary_content_and_reports_character_columns() {
        assert_eq!(text_source(b"plain text"), Some("plain text"));
        assert_eq!(text_source(b"binary\0text"), None);
        assert_eq!(text_source(&[0xff]), None);
        assert_eq!(location("first\nété target", "first\nété ".len()), (2, 5));
    }
}
