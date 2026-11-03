/* Independent GGML oracle for Q6_K fixtures. Link against the pinned llama.cpp
 * ggml-base library; no BML decoder participates in fixture generation. */
#include "ggml-quants.h"
#include <stdio.h>

int main(void) {
    block_q6_K block;
    float decoded[256];
    size_t count;
    while ((count = fread(&block, 1, sizeof block, stdin)) != 0) {
        if (count != sizeof block) return 2;
        dequantize_row_q6_K(&block, decoded, 256);
        if (fwrite(decoded, sizeof(float), 256, stdout) != 256) return 3;
    }
    return ferror(stdin) || fflush(stdout) != 0;
}
