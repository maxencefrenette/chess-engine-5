// One warp evaluates eight legal moves. The diagonal of Q Kᵀ contains
// the eight requested dot products. Two K=16 tiles cover the 32 channels.
extern "C" __global__ void legal_policy_dots(const unsigned short* queries,
                                             const unsigned short* keys,
                                             const unsigned int* indices,
                                             unsigned short* output,
                                             unsigned int count,
                                             unsigned int move_width) {
    const unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int first = (thread / 32) * 8;
    if (first >= count) return;
    const unsigned int row = (thread % 32) / 4;
    const unsigned int lane = thread % 4;
    const unsigned int index = first + row;
    // All lanes participate in MMA, even for a partial final tile.
    const unsigned int input = index < count ? index : first;
    const size_t board = (size_t)(input / move_width) * 64;
    const unsigned int move = indices[input];
    const unsigned short* query = queries + (board + move / 64) * 32;
    const unsigned short* key = keys + (board + move % 64) * 32;

    float even = 0, odd = 0, unused_even = 0, unused_odd = 0;
    for (unsigned int base = 0; base < 32; base += 16) {
        const unsigned int c = base + 2 * lane;
        const unsigned int a0 = query[c] | ((unsigned int)query[c + 1] << 16);
        const unsigned int a1 = query[c + 8] | ((unsigned int)query[c + 9] << 16);
        const unsigned int b0 = key[c] | ((unsigned int)key[c + 1] << 16);
        const unsigned int b1 = key[c + 8] | ((unsigned int)key[c + 9] << 16);
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%4,%5,%5}, {%6,%7}, {%0,%1,%2,%3};"
            : "+f"(even), "+f"(odd), "+f"(unused_even), "+f"(unused_odd)
            : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    }
    if (index < count && lane == row / 2) {
        unsigned short bits;
        asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(bits) : "f"((row % 2) ? odd : even));
        output[index] = bits;
    }
}
