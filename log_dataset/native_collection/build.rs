use std::{env, fs, path::PathBuf, process::Command};
#[allow(dead_code)]
#[path = "src/schema.rs"]
mod schema;
fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let source = root.join("log_dataset/nyanten");
    println!("cargo:rerun-if-changed={}", source.join("vendor").display());
    println!("cargo:rerun-if-changed=src/schema.rs");
    let mut header = String::from(
        "#pragma once\n#include <cstddef>\nnamespace fields {\nstruct Field { size_t offset; int dtype; };\n",
    );
    for field in schema::FIELDS {
        header += &format!(
            "inline constexpr Field {}{{{}, {}}};\n",
            field.name, field.offset, field.dtype as u8
        );
    }
    header += "}\n";
    fs::write(out.join("feature_fields.hpp"), header).unwrap();
    println!("cargo:rerun-if-changed=cpp");
    println!(
        "cargo:rerun-if-changed={}",
        source.join("analysis.hpp").display()
    );
    let mut objects = Vec::new();
    for entry in fs::read_dir(&source)
        .unwrap()
        .chain(fs::read_dir("cpp").unwrap())
    {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|x| x == "cpp") {
            println!("cargo:rerun-if-changed={}", path.display());
            let target = out.join(path.file_name().unwrap()).with_extension("o");
            assert!(
                Command::new("c++")
                    .args(["-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-c"])
                    .arg(&path)
                    .arg("-I")
                    .arg(source.join("vendor"))
                    .arg("-I")
                    .arg(&source)
                    .arg("-I")
                    .arg(&out)
                    .arg("-I")
                    .arg("cpp")
                    .arg("-o")
                    .arg(&target)
                    .status()
                    .unwrap()
                    .success()
            );
            objects.push(target);
        }
    }
    // ar replaces matching members but retains objects from renamed sources.
    // Recreate the archive so only this build's C++ objects are linked.
    let archive = out.join("libnyanten.a");
    if archive.exists() {
        fs::remove_file(&archive).unwrap();
    }
    assert!(
        Command::new("ar")
            .arg("crs")
            .arg(&archive)
            .args(objects)
            .status()
            .unwrap()
            .success()
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=nyanten");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
