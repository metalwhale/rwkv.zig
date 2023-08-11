use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
};
use tokenizers::Tokenizer;

// TODO: Find a way to return an instance of `Tokenizer` to C

#[no_mangle]
pub extern "C" fn encode(
    tokenizer_file_path: *const c_char,
    prompt: *const c_char,
    len: *mut usize,
) -> *const usize {
    let tokenizer = Tokenizer::from_file(safetify(tokenizer_file_path))
        .expect("Tokenizer file opening failed.");
    let encoding = tokenizer
        .encode(safetify(prompt), false)
        .expect("Encoding failed");
    let ids: Vec<usize> = encoding
        .get_ids()
        .to_owned()
        .iter()
        .map(|&i| i as usize)
        .collect();
    unsafe { *len = ids.len() };
    let ids_ptr = ids.as_ptr();
    std::mem::forget(ids);
    return ids_ptr;
}

#[no_mangle]
pub extern "C" fn decode(
    tokenizer_file_path: *const c_char,
    token: usize,
    len: *mut usize,
) -> *const c_char {
    let tokenizer = Tokenizer::from_file(safetify(tokenizer_file_path))
        .expect("Tokenizer file opening failed.");
    let vocab = CString::new(
        tokenizer
            .decode(vec![token as u32], true)
            .expect("Decoding failed"),
    )
    .expect("CString conversion failed.");
    unsafe { *len = vocab.as_bytes().len() };
    let vocab_ptr = vocab.as_ptr();
    std::mem::forget(vocab);
    return vocab_ptr;
}

fn safetify(c_string: *const c_char) -> String {
    // See: https://doc.rust-lang.org/std/ffi/struct.CStr.html#examples
    return String::from_utf8_lossy(unsafe { CStr::from_ptr(c_string) }.to_bytes()).to_string();
}
