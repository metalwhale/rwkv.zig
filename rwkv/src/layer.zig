const std = @import("std");

pub const Emb = struct {
    vocab_size: usize,
    weight: []const f32,

    pub fn init(vocab_size: usize, weight: []const f32) Emb {
        const emb = Emb{ .vocab_size = vocab_size, .weight = weight };
        return emb;
    }
};

pub const Block = struct {
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

    pub fn init(
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
        };
        return block;
    }
};

pub const LnOut = struct {
    weight: []const f32,
    bias: []const f32,

    pub fn init(weight: []const f32, bias: []const f32) LnOut {
        const ln_out = LnOut{ .weight = weight, .bias = bias };
        return ln_out;
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
