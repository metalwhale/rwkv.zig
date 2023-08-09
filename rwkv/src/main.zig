const std = @import("std");
const Rwkv = @import("model.zig").Rwkv;
const Tokenizer = @import("tokenizer.zig").Tokenizer;

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 4) {
        std.debug.print("Usage: {s} <model_file_path> <info_file_path> <tokenizer_file_path>\n", .{args[0]});
        std.os.exit(1);
    }
    // Read model
    const model_file_path = args[1];
    const info_file_path = args[2];
    const tokenizer_file_path = args[3];
    const rwkv = try Rwkv.init(allocator, model_file_path, info_file_path);
    defer rwkv.deinit();
    // Read tokenizer
    const tokenizer = try Tokenizer.init(allocator, tokenizer_file_path);
    defer tokenizer.deinit();
    const prompt_tokens = Tokenizer.encode();
    for (prompt_tokens) |token| {
        std.debug.print("{s}", .{tokenizer.decode(token)});
    }
}
