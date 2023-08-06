const std = @import("std");

pub fn sliceCast(comptime T: type, source: []const u8) []const T {
    const address = @intFromPtr(source.ptr);
    const target_len = source.len / @sizeOf(T);
    return @as([*]const T, @ptrFromInt(address))[0..target_len];
}
