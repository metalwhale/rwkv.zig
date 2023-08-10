const std = @import("std");
const layer = @import("layer.zig");
const util = @import("util.zig");
const State = @import("state.zig").State;
const Allocator = std.mem.Allocator;
const Emb = layer.Emb;
const Block = layer.Block;
const LnOut = layer.LnOut;
const Head = layer.Head;

const Info = struct {
    const Self = @This();
    parsed: std.json.Parsed(std.json.Value),

    fn init(allocator: Allocator, buffer: []const u8) !Info {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buffer, .{});
        const info = Info{ .parsed = parsed };
        return info;
    }

    fn deinit(self: Self) void {
        self.parsed.deinit();
    }
};

const Layers = struct {
    const Self = @This();
    emb: Emb,
    blocks: []Block,
    ln_out: LnOut,
    head: Head,
    allocator: Allocator,

    fn init(allocator: Allocator, info: Info, memory: []const u8) !Layers {
        var pos: usize = 0;
        const vocab_size = readVocabSize(info);
        const ffn_hidden_dim = readFfnHiddenDim(info);
        // `emb` layer
        const emb = Emb.init(
            vocab_size,
            readWeight(info, "emb.weight", memory, &pos),
        );
        // `block` layers
        const blocks = try allocator.alloc(Block, 24); // TODO: Find a way to dynamically obtain blocks length from model file
        for (blocks, 0..) |*block, i| {
            const names = [_][]const u8{
                "ln1.weight",
                "ln1.bias",
                "ln2.weight",
                "ln2.bias",
                "att.time_decay",
                "att.time_first",
                "att.time_mix_k",
                "att.time_mix_v",
                "att.time_mix_r",
                "att.key.weight",
                "att.value.weight",
                "att.receptance.weight",
                "att.output.weight",
                "ffn.time_mix_k",
                "ffn.time_mix_r",
                "ffn.key.weight",
                "ffn.receptance.weight",
                "ffn.value.weight",
                "ln0.weight",
                "ln0.bias",
            };
            const weights = try allocator.alloc(?[]const f32, names.len);
            defer allocator.free(weights);
            for (weights, names) |*weight, name| {
                const full_name = try std.fmt.allocPrint(allocator, "blocks.{d}.{s}", .{ i, name });
                defer allocator.free(full_name);
                const should_read = i == 0 or
                    (!std.mem.eql(u8, name, "ln0.weight") and !std.mem.eql(u8, name, "ln0.bias"));
                weight.* = if (should_read) readWeight(info, full_name, memory, &pos) else null;
            }
            block.* = Block.init(
                allocator,
                ffn_hidden_dim,
                weights[0].?,
                weights[1].?,
                weights[2].?,
                weights[3].?,
                weights[4].?,
                weights[5].?,
                weights[6].?,
                weights[7].?,
                weights[8].?,
                weights[9].?,
                weights[10].?,
                weights[11].?,
                weights[12].?,
                weights[13].?,
                weights[14].?,
                weights[15].?,
                weights[16].?,
                weights[17].?,
                weights[18],
                weights[19],
            );
        }
        // `ln_out` layer
        const ln_out = LnOut.init(
            allocator,
            readWeight(info, "ln_out.weight", memory, &pos),
            readWeight(info, "ln_out.bias", memory, &pos),
        );
        // `head` layer
        const head = Head.init(
            vocab_size,
            readWeight(info, "head.weight", memory, &pos),
        );
        const layers = Layers{ .emb = emb, .blocks = blocks, .ln_out = ln_out, .head = head, .allocator = allocator };
        return layers;
    }

    fn deinit(self: Self) void {
        self.allocator.free(self.blocks);
    }

    fn readWeight(info: Info, name: []const u8, memory: []const u8, pos: *usize) []const f32 {
        const start = pos.*;
        var size: usize = 1;
        for (info.parsed.value.object.get(name).?.array.items) |dim| {
            size *= @intCast(dim.integer);
        }
        pos.* += size * @sizeOf(f32); // Directly change the value of pos for convenience
        return util.sliceCast(f32, memory[start..pos.*]);
    }

    fn readVocabSize(info: Info) usize {
        return @intCast(info.parsed.value.object.get("emb.weight").?.array.items[0].integer);
    }

    fn readFfnHiddenDim(info: Info) usize {
        return @intCast(info.parsed.value.object.get("blocks.0.ffn.key.weight").?.array.items[0].integer);
    }
};

pub const Rwkv = struct {
    const Self = @This();
    info: Info,
    layers: Layers,
    memory: []align(std.mem.page_size) const u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, model_file_path: []const u8, info_file_path: []const u8) !Rwkv {
        // Read info
        const info_buffer = try std.fs.cwd().readFileAlloc(allocator, info_file_path, 1024 * 1024);
        defer allocator.free(info_buffer);
        const info = try Info.init(allocator, info_buffer);
        // Read model
        const file = try std.fs.cwd().openFile(model_file_path, .{});
        defer file.close();
        const file_size = (try file.stat()).size;
        const memory = try std.os.mmap(
            null,
            file_size,
            std.os.PROT.READ,
            std.os.MAP.PRIVATE,
            file.handle,
            0,
        );
        const layers = try Layers.init(allocator, info, memory);
        const rwkv = Rwkv{ .allocator = allocator, .info = info, .layers = layers, .memory = memory };
        return rwkv;
    }

    pub fn deinit(self: Self) void {
        self.info.deinit();
        self.layers.deinit();
        std.os.munmap(self.memory);
    }

    pub fn getBlocksCount(self: Self) usize {
        return self.layers.blocks.len;
    }

    pub fn getDim(self: Self) usize {
        return self.layers.emb.weight.len / self.layers.emb.vocab_size;
    }

    pub fn infer(self: Self, token: usize, state: State) ![]f32 {
        const x = self.layers.emb.embed(token); // (dim)
        const x1 = try self.layers.blocks[0].norm(x, .ln0); // (dim)
        defer self.allocator.free(x1);
        for (self.layers.blocks, state.block_states) |block, block_state| {
            const tx = try block.norm(x1, .ln1); // (dim)
            defer self.allocator.free(tx);
            const tdx = try block.time_mixing(tx, block_state);
            defer self.allocator.free(tdx);
            for (x1, tdx) |*x1i, tdxi| {
                x1i.* += tdxi;
            }
            const cx = try block.norm(x1, .ln2); // (dim)
            defer self.allocator.free(cx);
            const cdx = try block.channel_mixing(cx, block_state);
            defer self.allocator.free(cdx);
            for (x1, cdx) |*x1i, cdxi| {
                x1i.* += cdxi;
            }
        }
        const x2 = try self.layers.ln_out.norm(x1);
        defer self.allocator.free(x2);
        const x3 = try self.allocator.alloc(f32, self.layers.head.vocab_size);
        defer self.allocator.free(x3);
        for (x3, 0..) |*x3i, i| {
            x3i.* = 0;
            for (self.layers.head.weight[i * x2.len .. (i + 1) * x2.len], x2) |hj, x2j| {
                x3i.* += hj * x2j;
            }
        }
        const probs = softmax(self.allocator, x3);
        return probs;
    }
};

fn softmax(allocator: Allocator, x: []const f32) ![]f32 {
    const out = try allocator.alloc(f32, x.len);
    // find max value (for numerical stability)
    var max_x: f32 = x[0];
    for (x[1..]) |xi| {
        if (xi > max_x) {
            max_x = xi;
        }
    }
    // exp and sum
    var sum: f32 = 0.0;
    for (out, x) |*oi, xi| {
        oi.* = @exp(xi - max_x);
        sum += oi.*;
    }
    // normalize
    for (out) |*oi| {
        oi.* /= sum;
    }
    return out;
}
