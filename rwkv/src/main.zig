const std = @import("std");
const Rwkv = @import("model.zig").Rwkv;
const State = @import("state.zig").State;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Allocator = std.mem.Allocator;

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 5) {
        std.debug.print("Usage: {s} <model_file_path> <info_file_path> <tokenizer_file_path> <prompt>\n", .{args[0]});
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
    const prompt_tokens = Tokenizer.encode(tokenizer_file_path, args[4]);
    var next_token: usize = undefined;
    var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
    for (prompt_tokens) |token| {
        std.debug.print("{s}", .{Tokenizer.decode(tokenizer_file_path, token)});
        const probs = try rwkv.infer(token, state);
        defer allocator.free(probs);
        next_token = try sample(allocator, &prng, probs, 1.0, 0.85);
    }
    for (0..100) |_| {
        std.debug.print("{s}", .{Tokenizer.decode(tokenizer_file_path, next_token)});
        const probs = try rwkv.infer(next_token, state);
        defer allocator.free(probs);
        next_token = try sample(allocator, &prng, probs, 1.0, 0.85);
    }
}

fn sample(allocator: Allocator, prng: *std.rand.DefaultPrng, probs: []f32, temperature: f32, top_p: f32) !usize {
    // Find cutoff
    const buff_probs = try allocator.alloc(f32, probs.len);
    defer allocator.free(buff_probs);
    std.mem.copy(f32, buff_probs, probs);
    std.sort.insertion(f32, buff_probs, {}, std.sort.asc(f32));
    var sum: f32 = 0;
    var cutoff: f32 = 0;
    for (buff_probs) |p| {
        sum += p;
        if (sum > top_p) {
            cutoff = p;
            break;
        }
    }
    // Remove low probability tokens and recalculate probs
    std.mem.copy(f32, buff_probs, probs);
    sum = 0;
    for (buff_probs) |*p| {
        p.* = if (p.* < cutoff) 0 else std.math.pow(f32, p.*, 1 / temperature);
        sum += p.*;
    }
    for (buff_probs) |*p| {
        p.* /= sum;
    }
    // Random choice
    const r = prng.random().float(f32);
    sum = 0.0;
    for (buff_probs, 0..) |p, i| {
        sum += p;
        if (r < sum) {
            return i;
        }
    }
    // in case of rounding errors
    return buff_probs.len - 1;
}
