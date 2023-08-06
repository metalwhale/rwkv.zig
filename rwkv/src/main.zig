const std = @import("std");
const Rwkv = @import("model.zig").Rwkv;

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
    const rwkv = try Rwkv.init(allocator, model_file_path, info_file_path);
    defer rwkv.deinit();
}
