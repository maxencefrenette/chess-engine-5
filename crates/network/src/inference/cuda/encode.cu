extern "C" __global__ void encode_boards(const unsigned char* boards, float* output, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int channel = i % 21;
    unsigned int square = (i / 21) % 64;
    const unsigned char* b = boards + (i / (21 * 64)) * 37;
    unsigned int piece = (b[square / 2] >> (4 * (square % 2))) & 15;
    float value;
    if (channel < 13) value = channel == piece ? 1.0f : 0.0f;
    else if (channel == 13) value = (float)b[32];
    else if (channel < 18) value = (float)((b[33] >> (channel - 14)) & 1);
    else if (channel == 18) value = b[34] == square ? 1.0f : 0.0f;
    else if (channel == 19) value = __fdiv_rn((float)b[35], 100.0f);
    else value = (float)b[36] - 1.0f;
    output[i] = value;
}
