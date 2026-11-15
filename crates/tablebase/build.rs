use std::{fs, path::Path};

fn sources(build: &mut cc::Build, directory: &str, extension: &str) {
    let mut paths: Vec<_> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == extension))
        .collect();
    paths.sort();
    build.files(paths);
}

fn main() {
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-changed=probe.cpp");
    let root = Path::new("vendor/chesstb");
    assert!(
        root.join("src/probe/probe.h").is_file(),
        "ChessTB submodule is missing; run git submodule update --init --recursive"
    );
    let mut c = cc::Build::new();
    c.include(root.join("lib")).opt_level(3).warnings(false);
    for dir in [
        "lz4",
        "LZMA",
        "zstd/common",
        "zstd/compress",
        "zstd/dictBuilder",
    ] {
        sources(&mut c, &format!("vendor/chesstb/lib/{dir}"), "c");
    }
    c.compile("chesstb_compression");
    let mut cpp = cc::Build::new();
    cpp.cpp(true)
        .std("c++17")
        .include(root.join("src"))
        .include(root.join("lib"))
        .define("NDEBUG", None)
        .opt_level(3)
        .warnings(false)
        .file("probe.cpp");
    for dir in ["chess", "probe", "util"] {
        sources(&mut cpp, &format!("vendor/chesstb/src/{dir}"), "cpp");
    }
    cpp.compile("chesstb_probe");
}
