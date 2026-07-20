use ignore::WalkBuilder;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;

const EXTS: [&str; 4] = ["gguf", "safetensors", "ckpt", "pt"];
const BUF_SIZE: usize = 8 * 1024 * 1024; // 8MiB read chunks

#[derive(Serialize)]
struct ModelHash {
    original_path: String,
    realpath: String,
    sha256: String,
    size: u64,
    size_human: String,
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{}{}", bytes, UNITS[0])
    } else if (size - size.round()).abs() < 0.05 {
        format!("{}{}", size.round() as u64, UNITS[unit])
    } else {
        format!("{:.1}{}", size, UNITS[unit])
    }
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(BUF_SIZE, file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; BUF_SIZE];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn process(path: &Path) -> Option<ModelHash> {
    let meta = fs::symlink_metadata(path).ok()?;
    // follow symlinks for size/hash like `stat -L` / `sha256sum` do
    let real_meta = fs::metadata(path).unwrap_or(meta);
    let size = real_meta.len();

    let realpath = fs::canonicalize(path)
        .ok()?
        .to_string_lossy()
        .into_owned();

    let sha256 = match sha256_file(path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("skip {}: {}", path.display(), e);
            return None;
        }
    };

    Some(ModelHash {
        original_path: path.to_string_lossy().into_owned(),
        realpath,
        sha256,
        size,
        size_human: human_size(size),
    })
}

fn main() {
    let search_path = env::args().nth(1).unwrap_or_else(|| ".".to_string());

    // Collect candidate files first (fast, single-threaded walk; -I -H equivalent:
    // don't respect .gitignore, do include hidden files).
    let mut paths = Vec::new();
    let walker = WalkBuilder::new(&search_path)
        .hidden(false)       // include hidden (-H)
        .ignore(false)       // don't respect .ignore
        .git_ignore(false)   // don't respect .gitignore (-I)
        .git_global(false)
        .git_exclude(false)
        .build();

    for entry in walker.filter_map(|e| e.ok()) {
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if EXTS.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                paths.push(path.to_path_buf());
            }
        }
    }

    // Hash in parallel. rayon defaults to num_cpus threads; override with
    // RAYON_NUM_THREADS=N if you want to leave headroom for NVMe queue depth
    // vs. CPU-bound SHA-256 (usually not necessary — SHA-NI is fast enough
    // that this becomes I/O-bound quickly).
    //
    // Perf testing on my system did not reveal any gains using more or less threads, 
    // so we just use the default global pool. YMMV
    let mut results: Vec<ModelHash> = paths.par_iter().filter_map(|p| process(p)).collect();

    results.sort_by(|a, b| a.original_path.cmp(&b.original_path));

    let json = serde_json::to_string_pretty(&results).expect("serialize failed");
    fs::write("model-hashes.json", json).expect("failed to write model-hashes.json");

    println!("Generated model-hashes.json ({} files)", results.len());
}