const std = @import("std");
const Rwkv = @import("model.zig").Rwkv;
const State = @import("state.zig").State;
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
    // Inference
    const state = try State.init(allocator, rwkv.getBlocksCount(), rwkv.getDim());
    defer state.deinit();
    const prompt_tokens = Tokenizer.encode(tokenizer_file_path, "Hello darkness, my old friend.");
    var next_token: usize = undefined;
    for (prompt_tokens) |token| {
        std.debug.print("{s}", .{Tokenizer.decode(tokenizer_file_path, token)});
        const probs = try rwkv.infer(token, state);
        defer allocator.free(probs);
        next_token = sample(probs);
    }
    for (0..20) |_| {
        std.debug.print("{s}", .{Tokenizer.decode(tokenizer_file_path, next_token)});
        const probs = try rwkv.infer(next_token, state);
        defer allocator.free(probs);
        next_token = sample(probs);
    }
}

fn sample(probs: []f32) usize {
    // TODO: Choose token randomly
    var token: usize = 0;
    var max_p = probs[token];
    for (probs[1..], 1..) |p, i| {
        if (p > max_p) {
            max_p = p;
            token = i;
        }
    }
    return token;
}
