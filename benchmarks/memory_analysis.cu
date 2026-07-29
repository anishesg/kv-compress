#include "codebook.cuh"

#include <cstdio>
#include <cmath>
#include <string>

// Compute memory usage in bytes for a single head at a given sequence length.
//
// Full-precision: 2 (K+V) * seq_len * head_dim * sizeof(fp16)
// Mixed:          2 * age_threshold * head_dim * sizeof(fp16)  [full-precision region]
//               + 2 * max(0, seq_len - age_threshold) * sizeof(uint8)  [compressed region]
//
// The codebook itself contributes a fixed overhead of
//   2 * codebook_size * head_dim * sizeof(float32)
// shared across all tokens.

struct MemStats {
    double full_bytes;        // full-precision KV cache bytes (one head)
    double mixed_bytes;       // mixed KV cache bytes including codebook
    double codebook_bytes;    // fixed codebook overhead
    double compressed_bytes;  // variable compressed region
    double full_region_bytes; // variable full-precision region
    double ratio;             // full / mixed compression ratio
};

static MemStats compute(int seq_len, int head_dim, int codebook_size, int age_threshold) {
    MemStats s{};
    s.full_bytes = 2.0 * seq_len * head_dim * sizeof(__half);

    int full_region_tokens = (seq_len < age_threshold) ? seq_len : age_threshold;
    int comp_tokens        = (seq_len > age_threshold) ? (seq_len - age_threshold) : 0;

    s.full_region_bytes = 2.0 * full_region_tokens * head_dim * sizeof(__half);
    s.compressed_bytes  = 2.0 * comp_tokens * sizeof(uint8_t);
    s.codebook_bytes    = 2.0 * codebook_size * head_dim * sizeof(float);

    s.mixed_bytes = s.full_region_bytes + s.compressed_bytes + s.codebook_bytes;
    s.ratio       = (s.mixed_bytes > 0) ? (s.full_bytes / s.mixed_bytes) : 0.0;
    return s;
}

static const char* human_bytes(double b) {
    static char buf[32];
    if (b >= 1e9)       snprintf(buf, sizeof(buf), "%7.2f GB", b / 1e9);
    else if (b >= 1e6)  snprintf(buf, sizeof(buf), "%7.2f MB", b / 1e6);
    else if (b >= 1e3)  snprintf(buf, sizeof(buf), "%7.2f KB", b / 1e3);
    else                snprintf(buf, sizeof(buf), "%7.0f  B", b);
    return buf;
}

int main() {
    const int head_dim      = 128;
    const int codebook_size = 256;
    const int num_heads     = 32;   // representative 7B-class model
    const int age_threshold = 512;

    printf("Memory footprint analysis\n");
    printf("  head_dim=%d  codebook_size=%d  age_threshold=%d  num_heads=%d\n\n",
           head_dim, codebook_size, age_threshold, num_heads);

    printf("%-10s  %-12s  %-12s  %-12s  %-12s  %-12s  %-8s\n",
           "seq_len",
           "full(1head)",
           "mixed(1head)",
           "full(32head)",
           "mixed(32head)",
           "saved",
           "ratio");
    printf("%s\n", std::string(92, '-').c_str());

    int seq_lengths[] = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    };

    for (int seq_len : seq_lengths) {
        MemStats s = compute(seq_len, head_dim, codebook_size, age_threshold);

        double full_total   = s.full_bytes  * num_heads;
        double mixed_total  = s.mixed_bytes * num_heads;
        double saved        = full_total - mixed_total;

        printf("%-10d  %-12s  %-12s  %-12s  %-12s  %-12s  %7.1fx\n",
               seq_len,
               human_bytes(s.full_bytes),
               human_bytes(s.mixed_bytes),
               human_bytes(full_total),
               human_bytes(mixed_total),
               human_bytes(saved),
               s.ratio);
    }

    printf("\n");
    printf("Breakdown at seq_len=131072 (128K tokens):\n");
    {
        MemStats s = compute(131072, head_dim, codebook_size, age_threshold);
        printf("  Full-precision region  : %s  (%d tokens * %d dim * 2 bytes * 2 KV)\n",
               human_bytes(s.full_region_bytes), age_threshold, head_dim);
        printf("  Compressed region      : %s  (%d tokens * 1 byte * 2 KV)\n",
               human_bytes(s.compressed_bytes), 131072 - age_threshold);
        printf("  Codebook overhead      : %s  (%d entries * %d dim * 4 bytes * 2 KV)\n",
               human_bytes(s.codebook_bytes), codebook_size, head_dim);
        printf("  Total mixed (1 head)   : %s\n",
               human_bytes(s.mixed_bytes));
        printf("  Total full  (1 head)   : %s\n",
               human_bytes(s.full_bytes));
        printf("  Compression ratio      : %.1fx\n", s.ratio);
        printf("\n");
        printf("  Total mixed (32 heads) : %s\n", human_bytes(s.mixed_bytes * num_heads));
        printf("  Total full  (32 heads) : %s\n", human_bytes(s.full_bytes * num_heads));
        printf("  Memory saved           : %s\n",
               human_bytes((s.full_bytes - s.mixed_bytes) * num_heads));
    }

    printf("\n");
    printf("Effect of varying age_threshold at seq_len=131072:\n");
    printf("%-15s  %-12s  %-12s  %-8s\n",
           "age_threshold", "mixed(1head)", "full(1head)", "ratio");
    printf("%s\n", std::string(52, '-').c_str());
    for (int at : {128, 256, 512, 1024, 2048, 4096}) {
        MemStats s = compute(131072, head_dim, codebook_size, at);
        printf("%-15d  %-12s  %-12s  %7.1fx\n",
               at, human_bytes(s.mixed_bytes), human_bytes(s.full_bytes), s.ratio);
    }

    return 0;
}
