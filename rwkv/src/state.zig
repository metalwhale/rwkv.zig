const std = @import("std");
const Allocator = std.mem.Allocator;

pub const State = struct {
    pub const BlockState = struct {
        time_x: []f32, // (dim)
        time_num: []f32, // (dim)
        time_den: []f32, // (dim)
        channel_x: []f32, // (dim)
    };
    const Self = @This();
    block_states: []BlockState,
    allocator: Allocator,

    pub fn init(allocator: Allocator, blocks_count: usize, dim: usize) !State {
        const block_states = try allocator.alloc(BlockState, blocks_count);
        for (block_states) |*block_state| {
            const time_x = try allocator.alloc(f32, dim);
            const time_num = try allocator.alloc(f32, dim);
            const time_den = try allocator.alloc(f32, dim);
            const channel_x = try allocator.alloc(f32, dim);
            for (time_x, time_num, time_den, channel_x) |*tx, *tn, *td, *cx| {
                tx.* = 0.0;
                tn.* = 0.0;
                td.* = 0.0;
                cx.* = 0.0;
            }
            block_state.* = BlockState{ .time_x = time_x, .time_num = time_num, .time_den = time_den, .channel_x = channel_x };
        }
        const state = State{ .block_states = block_states, .allocator = allocator };
        return state;
    }

    pub fn deinit(self: Self) void {
        for (self.block_states) |s| {
            self.allocator.free(s.time_x);
            self.allocator.free(s.time_num);
            self.allocator.free(s.time_den);
            self.allocator.free(s.channel_x);
        }
        self.allocator.free(self.block_states);
    }
};
