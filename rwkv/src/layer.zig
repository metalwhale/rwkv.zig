const std = @import("std");
const State = @import("state.zig").State;
const Allocator = std.mem.Allocator;

pub const Emb = struct {
    const Self = @This();
    vocab_size: usize,
    weight: []const f32,

    pub fn init(vocab_size: usize, weight: []const f32) Emb {
        const emb = Emb{ .vocab_size = vocab_size, .weight = weight };
        return emb;
    }

    pub fn embed(self: Self, token: usize) []const f32 {
        const dim = self.weight.len / self.vocab_size;
        return self.weight[token * dim .. (token + 1) * dim];
    }
};

pub const Block = struct {
    pub const NormType = enum {
        ln0,
        ln1,
        ln2,
    };
    const Self = @This();
    ffn_hidden_dim: usize,
    ln1_weight: []const f32,
    ln1_bias: []const f32,
    ln2_weight: []const f32,
    ln2_bias: []const f32,
    att_time_decay: []const f32,
    att_time_first: []const f32,
    att_time_mix_k: []const f32,
    att_time_mix_v: []const f32,
    att_time_mix_r: []const f32,
    att_key_weight: []const f32,
    att_value_weight: []const f32,
    att_receptance_weight: []const f32,
    att_output_weight: []const f32,
    ffn_time_mix_k: []const f32,
    ffn_time_mix_r: []const f32,
    ffn_key_weight: []const f32,
    ffn_receptance_weight: []const f32,
    ffn_value_weight: []const f32,
    ln0_weight: ?[]const f32 = null,
    ln0_bias: ?[]const f32 = null,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        ffn_hidden_dim: usize,
        ln1_weight: []const f32,
        ln1_bias: []const f32,
        ln2_weight: []const f32,
        ln2_bias: []const f32,
        att_time_decay: []const f32,
        att_time_first: []const f32,
        att_time_mix_k: []const f32,
        att_time_mix_v: []const f32,
        att_time_mix_r: []const f32,
        att_key_weight: []const f32,
        att_value_weight: []const f32,
        att_receptance_weight: []const f32,
        att_output_weight: []const f32,
        ffn_time_mix_k: []const f32,
        ffn_time_mix_r: []const f32,
        ffn_key_weight: []const f32,
        ffn_receptance_weight: []const f32,
        ffn_value_weight: []const f32,
        ln0_weight: ?[]const f32,
        ln0_bias: ?[]const f32,
    ) Block {
        const block = Block{
            .ffn_hidden_dim = ffn_hidden_dim,
            .ln1_weight = ln1_weight,
            .ln1_bias = ln1_bias,
            .ln2_weight = ln2_weight,
            .ln2_bias = ln2_bias,
            .att_time_decay = att_time_decay,
            .att_time_first = att_time_first,
            .att_time_mix_k = att_time_mix_k,
            .att_time_mix_v = att_time_mix_v,
            .att_time_mix_r = att_time_mix_r,
            .att_key_weight = att_key_weight,
            .att_value_weight = att_value_weight,
            .att_receptance_weight = att_receptance_weight,
            .att_output_weight = att_output_weight,
            .ffn_time_mix_k = ffn_time_mix_k,
            .ffn_time_mix_r = ffn_time_mix_r,
            .ffn_key_weight = ffn_key_weight,
            .ffn_receptance_weight = ffn_receptance_weight,
            .ffn_value_weight = ffn_value_weight,
            .ln0_weight = ln0_weight,
            .ln0_bias = ln0_bias,
            .allocator = allocator,
        };
        return block;
    }

    pub fn norm(self: Self, x: []const f32, norm_type: NormType) ![]f32 {
        return switch (norm_type) {
            .ln0 => try layer_norm(self.allocator, x, self.ln0_weight.?, self.ln0_bias.?),
            .ln1 => try layer_norm(self.allocator, x, self.ln1_weight, self.ln1_bias),
            .ln2 => try layer_norm(self.allocator, x, self.ln2_weight, self.ln2_bias),
        };
    }

    pub fn time_mixing(self: Self, x: []const f32, block_state: State.BlockState) ![]f32 {
        const decay = self.att_time_decay; // (dim)
        const bonus = self.att_time_first; // (dim)
        const mix_k = self.att_time_mix_k; // (dim)
        const mix_v = self.att_time_mix_v; // (dim)
        const mix_r = self.att_time_mix_r; // (dim)
        const Wk = self.att_key_weight; // (W_dim, dim)
        const Wv = self.att_value_weight; // (W_dim, dim)
        const Wr = self.att_receptance_weight; // (W_dim, dim)
        const Wout = self.att_output_weight; // (W_dim, dim)
        const dim = x.len;
        const W_dim = Wk.len / dim;
        if (W_dim != dim) {
            unreachable; // TODO: Confirm if it is necessary for Ws to have shapes of (W_dim, dim) with W_dim = dim
        }
        const k = try self.allocator.alloc(f32, W_dim);
        const v = try self.allocator.alloc(f32, W_dim);
        const r = try self.allocator.alloc(f32, W_dim);
        defer self.allocator.free(k);
        defer self.allocator.free(v);
        defer self.allocator.free(r);
        for (k, v, r, 0..) |*ki, *vi, *ri, i| {
            ki.* = 0;
            vi.* = 0;
            ri.* = 0;
            for (
                Wk[i * dim .. (i + 1) * dim],
                Wv[i * dim .. (i + 1) * dim],
                Wr[i * dim .. (i + 1) * dim],
                x,
                block_state.time_x,
                mix_k,
                mix_v,
                mix_r,
            ) |Wkj, Wvj, Wrj, xj, txj, mkj, mvj, mrj| {
                ki.* += Wkj * (xj * mkj + txj * (1 - mkj));
                vi.* += Wvj * (xj * mvj + txj * (1 - mvj));
                ri.* += Wrj * (xj * mrj + txj * (1 - mrj));
            }
        }
        const wkv = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(wkv);
        for (wkv, block_state.time_num, block_state.time_den, bonus, k, v) |*wkvi, tni, tdi, bi, ki, vi| {
            wkvi.* = (tni + @exp(bi + ki) * vi) / (tdi + @exp(bi + ki));
        }
        const rwkv = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(rwkv);
        sigmoid(r);
        for (rwkv, r, wkv) |*rwkvi, ri, wkvi| {
            rwkvi.* = ri * wkvi;
        }
        for (block_state.time_x, block_state.time_num, block_state.time_den, x, decay, k, v) |*txi, *tni, *tdi, xi, di, ki, vi| {
            txi.* = xi;
            tni.* = @exp(-@exp(di)) * tni.* + @exp(ki) * vi;
            tdi.* = @exp(-@exp(di)) * tdi.* + @exp(ki);
        }
        const dx = try self.allocator.alloc(f32, dim);
        for (dx, 0..) |*dxi, i| {
            dxi.* = 0;
            for (Wout[i * dim .. (i + 1) * dim], rwkv) |Woj, rwkvj| {
                dxi.* += Woj * rwkvj;
            }
        }
        return dx;
    }

    pub fn channel_mixing(self: Self, x: []const f32, block_state: State.BlockState) ![]f32 {
        const mix_k = self.ffn_time_mix_k;
        const mix_r = self.ffn_time_mix_r;
        const Wk = self.ffn_key_weight;
        const Wr = self.ffn_receptance_weight;
        const Wv = self.ffn_value_weight;
        const dim = x.len;
        const Wk_dim = Wk.len / dim;
        const Wr_dim = Wr.len / dim;
        const k = try self.allocator.alloc(f32, Wk_dim);
        const r = try self.allocator.alloc(f32, Wr_dim);
        defer self.allocator.free(k);
        defer self.allocator.free(r);
        for (k, 0..) |*ki, i| {
            ki.* = 0;
            for (Wk[i * dim .. (i + 1) * dim], x, block_state.channel_x, mix_k) |Wkj, xj, cxj, mkj| {
                ki.* += Wkj * (xj * mkj + cxj * (1 - mkj));
            }
        }
        for (r, 0..) |*ri, i| {
            ri.* = 0;
            for (Wr[i * dim .. (i + 1) * dim], x, block_state.channel_x, mix_r) |Wrj, xj, cxj, mrj| {
                ri.* += Wrj * (xj * mrj + cxj * (1 - mrj));
            }
        }
        const Wv_dim = Wv.len / dim;
        const vk = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(vk);
        for (vk, 0..) |*vki, i| {
            vki.* = 0;
            for (Wv[i * Wv_dim .. (i + 1) * Wv_dim], k) |Wvj, kj| {
                vki.* += Wvj * std.math.pow(f32, @max(0, kj), 2);
            }
        }
        for (block_state.channel_x, x) |*cxi, xi| {
            cxi.* = xi;
        }
        sigmoid(r);
        const dx = try self.allocator.alloc(f32, dim);
        for (dx, r, vk) |*dxi, ri, vki| {
            dxi.* = ri * vki;
        }
        return dx;
    }
};

pub const LnOut = struct {
    const Self = @This();
    weight: []const f32,
    bias: []const f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, weight: []const f32, bias: []const f32) LnOut {
        const ln_out = LnOut{ .weight = weight, .bias = bias, .allocator = allocator };
        return ln_out;
    }

    pub fn norm(self: Self, x: []const f32) ![]f32 {
        return try layer_norm(self.allocator, x, self.weight, self.bias);
    }
};

pub const Head = struct {
    vocab_size: usize,
    weight: []const f32,

    pub fn init(vocab_size: usize, weight: []const f32) Head {
        const head = Head{ .vocab_size = vocab_size, .weight = weight };
        return head;
    }
};

fn layer_norm(allocator: Allocator, x: []const f32, w: []const f32, b: []const f32) ![]f32 {
    if (x.len != w.len or x.len != b.len) {
        unreachable;
    }
    const out = try allocator.alloc(f32, x.len);
    const mx = mean(x);
    const sx = sd(x);
    for (out, x, w, b) |*o, xi, wi, bi| {
        o.* = (xi - mx) / sx * wi + bi;
    }
    return out;
}

fn mean(x: []const f32) f32 {
    var s: f32 = 0.0;
    for (x) |xi| {
        s += xi;
    }
    const m = s / @as(f32, @floatFromInt(x.len));
    return m;
}

// Standard deviation
fn sd(x: []const f32) f32 {
    var m = mean(x);
    var v: f32 = 0.0; // variance
    for (x) |xi| {
        v += std.math.pow(f32, xi - m, 2);
    }
    v = v / @as(f32, @floatFromInt(x.len));
    const s = @sqrt(v);
    return s;
}

fn sigmoid(x: []f32) void {
    for (x) |*xi| {
        xi.* = 1 / (1 + @exp(-xi.*));
    }
}
