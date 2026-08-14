use anyhow::Result;
use std::process::Command;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo xtask <command>");
        println!("Commands: gen-header, build-android, bench");
        return Ok(());
    }
    match args[1].as_str() {
        "gen-header" => gen_header()?,
        "build-android" => build_android()?,
        "bench" => run_bench()?,
        _ => eprintln!("Unknown command"),
    }
    Ok(())
}

fn gen_header() -> Result<()> {
    cbindgen::generate("crates/hw-ffi")
        .expect("Unable to generate bindings")
        .write_to_file("hw-ffi.h");
    println!("Generated hw-ffi.h");
    Ok(())
}

fn build_android() -> Result<()> {
    // Use cargo-ndk to build for armeabi-v7a, arm64-v8a, x86_64.
    let targets = vec!["armv7-linux-androideabi", "aarch64-linux-android", "x86_64-linux-android"];
    for target in targets {
        let status = Command::new("cargo")
            .args(&["ndk", "-t", target, "build", "-p", "hw-ffi", "--release"])
            .status()?;
        if !status.success() {
            anyhow::bail!("Build failed for {}", target);
        }
    }
    println!("Android builds successful.");
    Ok(())
}

fn run_bench() -> Result<()> {
    let status = Command::new("cargo")
        .args(&["bench", "--workspace"])
        .status()?;
    if !status.success() {
        anyhow::bail!("Benchmarks failed");
    }
    Ok(())
}