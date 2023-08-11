const std = @import("std");
const tokenizers = @cImport(@cInclude("tokenizers.h"));

pub const Tokenizer = struct {
    pub fn encode(file_path: []const u8, prompt: []const u8) []const usize {
        var len: usize = 0;
        const ids = tokenizers.encode(file_path.ptr, prompt.ptr, &len);
        return ids[0..len];
    }

    pub fn decode(file_path: []const u8, token: usize) []const u8 {
        var len: usize = 0;
        const vocab = tokenizers.decode(file_path.ptr, token, &len);
        return vocab[0..len];
    }
};
