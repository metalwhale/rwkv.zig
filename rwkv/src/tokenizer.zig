const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Tokenizer = struct {
    const Self = @This();
    vocabs: [][]u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, file_path: []const u8) !Tokenizer {
        const buffer = try std.fs.cwd().readFileAlloc(allocator, file_path, 3 * 1024 * 1024);
        defer allocator.free(buffer);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buffer, .{});
        defer parsed.deinit();
        const vocab_size = 50277; // TODO: Calculate from `added_tokens` and `vocab_map`
        const vocabs = try allocator.alloc([]u8, vocab_size);
        const added_tokens = parsed.value.object.get("added_tokens").?.array;
        for (added_tokens.items) |item| {
            const token: usize = @intCast(item.object.get("id").?.integer);
            const token_str = item.object.get("content").?.string;
            vocabs[token] = try convert(allocator, token_str);
        }
        const vocab_map = parsed.value.object.get("model").?.object.get("vocab").?.object;
        var vocab_iterator = vocab_map.iterator();
        while (vocab_iterator.next()) |entry| {
            const token: usize = @intCast(entry.value_ptr.*.integer);
            const token_str = entry.key_ptr.*;
            vocabs[token] = try convert(allocator, token_str);
        }
        const tokenizer = Tokenizer{ .vocabs = vocabs, .allocator = allocator };
        return tokenizer;
    }

    pub fn deinit(self: Self) void {
        for (self.vocabs) |token_str| {
            self.allocator.free(token_str);
        }
        self.allocator.free(self.vocabs);
    }

    pub fn encode() []const usize {
        // TODO: Implement true tokenizer
        return &[_]usize{ 12092, 13862, 13, 619, 1711, 3331, 15 }; // "Hello darkness, my old friend."
    }

    pub fn decode(self: Self, token: usize) []u8 {
        return self.vocabs[token];
    }
};

// See: https://github.com/huggingface/tokenizers/blob/v0.13.3/tokenizers/src/pre_tokenizers/byte_level.rs#L153
// TODO: Fully implement `decode_chain`, not only handling \u0120
fn convert(allocator: Allocator, source: []const u8) ![]u8 {
    // Replace \u0120 with space
    if (source[0] == 196 and source[1] == 160) { // UTF-8 'Ġ' (\u0120)
        const target = try allocator.alloc(u8, source.len - 1);
        target[0] = 32; // UTF-8 'space' (\u0020)
        std.mem.copy(u8, target[1..], source[2..]);
        return target;
    } else {
        const target = try allocator.alloc(u8, source.len);
        std.mem.copy(u8, target, source);
        return target;
    }
}
