use tch::{Kind, Tensor};

mod test_utils;
use test_utils::*;

struct TmpFile(std::path::PathBuf);

impl TmpFile {
    fn create(base: &str) -> TmpFile {
        let filename = std::env::temp_dir().join(format!(
            "tch-{}-{}-{:?}",
            base,
            std::process::id(),
            std::thread::current().id(),
        ));
        TmpFile(filename)
    }
}

impl std::convert::AsRef<std::path::Path> for TmpFile {
    fn as_ref(&self) -> &std::path::Path {
        self.0.as_path()
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap()
    }
}

#[test]
fn save_and_load() {
    let tmp_file = TmpFile::create("save-and-load");
    let vec = [3.0, 1.0, 4.0, 1.0, 5.0].to_vec();
    let t1 = Tensor::from_slice(&vec);
    t1.save(&tmp_file).unwrap();
    let t2 = Tensor::load(&tmp_file).unwrap();
    assert_eq!(vec_f64_from(&t2), vec)
}

#[test]
fn save_to_stream_and_load() {
    let tmp_file = TmpFile::create("write-stream");
    let vec = [3.0, 1.0, 4.0, 1.0, 5.0].to_vec();
    let t1 = Tensor::from_slice(&vec);
    t1.save_to_stream(std::fs::File::create(&tmp_file).unwrap()).unwrap();
    let t2 = Tensor::load(&tmp_file).unwrap();
    assert_eq!(vec_f64_from(&t2), vec)
}

#[test]
fn save_and_load_from_stream() {
    let tmp_file = TmpFile::create("read-stream");
    let vec = [3.0, 1.0, 4.0, 1.0, 5.0].to_vec();
    let t1 = Tensor::from_slice(&vec);
    t1.save(&tmp_file).unwrap();
    let reader = std::io::BufReader::new(std::fs::File::open(&tmp_file).unwrap());
    let t2 = Tensor::load_from_stream(reader).unwrap();
    assert_eq!(vec_f64_from(&t2), vec)
}

#[test]
fn save_and_load_multi() {
    let tmp_file = TmpFile::create("save-and-load-multi");
    let pi = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::from_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::save_multi(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::load_multi(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(from::<i64>(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_to_stream_and_load_multi() {
    let tmp_file = TmpFile::create("save-to-stream-and-load-multi");
    let pi = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::from_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::save_multi_to_stream(
        &[(&"pi", &pi), (&"e", &e)],
        std::fs::File::create(&tmp_file).unwrap(),
    )
    .unwrap();
    let named_tensors = Tensor::load_multi(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(from::<i64>(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_and_load_multi_from_stream() {
    let tmp_file = TmpFile::create("save-and-load-multi-from-stream");
    let pi = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::from_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::save_multi(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let reader = std::io::BufReader::new(std::fs::File::open(&tmp_file).unwrap());
    let named_tensors = Tensor::load_multi_from_stream(reader).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(from::<i64>(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_and_load_npz() {
    let tmp_file = TmpFile::create("save-and-load-npz");
    let pi = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::from_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::write_npz(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::read_npz(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(from::<i64>(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn npz_rejects_conflicting_local_filenames_before_reading_tensors() {
    use std::io::{Cursor, Write};

    let tmp_file = TmpFile::create("npz-conflicting-filenames");
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for name in ["first.npy", "second.npy"] {
        writer.start_file(name, options).unwrap();
        writer.write_all(b"invalid tensor payload").unwrap();
    }
    let mut bytes = writer.finish().unwrap().into_inner();
    let local_header = zip::ZipArchive::new(Cursor::new(&bytes))
        .unwrap()
        .by_name("second.npy")
        .unwrap()
        .header_start() as usize;
    // Keep the central directory intact and change only the second local filename.
    bytes[local_header + 30] = b'x';
    std::fs::write(&tmp_file, bytes).unwrap();

    let error = Tensor::read_npz(&tmp_file).unwrap_err();
    assert!(error.to_string().contains("filename mismatch"), "{error}");
}

#[test]
fn save_and_load_npz_half() {
    let tmp_file = TmpFile::create("save-and-load-npz-half");
    let pi = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]).to_dtype(Kind::Half, true, false);
    let e =
        Tensor::from_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]).to_dtype(Kind::Half, true, false);
    Tensor::write_npz(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::read_npz(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(from::<i64>(&named_tensors[1].1.sum(tch::Kind::Float)), 57);
}

#[test]
fn save_and_load_npz_byte() {
    let tmp_file = TmpFile::create("save-and-load-npz-byte");
    let pi = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]).to_dtype(Kind::Int8, true, false);
    let e =
        Tensor::from_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]).to_dtype(Kind::Int8, true, false);
    Tensor::write_npz(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::read_npz(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    assert_eq!(named_tensors[0].0, "pi");
    assert_eq!(named_tensors[1].0, "e");
    assert_eq!(from::<i8>(&named_tensors[1].1.sum(tch::Kind::Int8)), 57);
}

#[test]
fn save_and_load_npy() {
    let tmp_file = TmpFile::create("save-and-load-npy");
    let pi = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    pi.write_npy(&tmp_file).unwrap();
    let pi = Tensor::read_npy(&tmp_file).unwrap();
    assert_eq!(vec_f64_from(&pi), [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    let pi = pi.reshape([3, 1, 2]);
    pi.write_npy(&tmp_file).unwrap();
    let pi = Tensor::read_npy(&tmp_file).unwrap();
    assert_eq!(pi.size(), [3, 1, 2]);
    assert_eq!(vec_f64_from(&pi.flatten(0, -1)), [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
}

#[test]
fn save_and_load_safetensors() {
    let tmp_file = TmpFile::create("save-and-load-safetensors");
    let pi = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]);
    let e = Tensor::from_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]);
    Tensor::write_safetensors(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::read_safetensors(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    for (name, tensor) in named_tensors {
        match name.as_str() {
            "pi" => assert_eq!(from::<i64>(&tensor.sum(tch::Kind::Float)), 14),
            "e" => assert_eq!(from::<i64>(&tensor.sum(tch::Kind::Float)), 57),
            _ => panic!("unknow name tensors"),
        }
    }
}

#[test]
fn save_and_load_safetensors_half() {
    let tmp_file = TmpFile::create("save-and-load-safetensors-half");
    let pi = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0]).to_dtype(Kind::Half, true, false);
    let e =
        Tensor::from_slice(&[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 6]).to_dtype(Kind::Half, true, false);
    Tensor::write_safetensors(&[(&"pi", &pi), (&"e", &e)], &tmp_file).unwrap();
    let named_tensors = Tensor::read_safetensors(&tmp_file).unwrap();
    assert_eq!(named_tensors.len(), 2);
    for (name, tensor) in named_tensors {
        match name.as_str() {
            "pi" => assert_eq!(from::<i64>(&tensor.sum(tch::Kind::Float)), 14),
            "e" => assert_eq!(from::<i64>(&tensor.sum(tch::Kind::Float)), 57),
            _ => panic!("unknow name tensors"),
        }
    }
}
