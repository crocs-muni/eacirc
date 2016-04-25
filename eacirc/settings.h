#pragma once

#include <core/base.h>
#include <string>

namespace settings {
struct Main {
    bool enable_cuda;
    std::string project;
    std::string evaluator;
    unsigned evaluator_precision;
    unsigned significance_level;
};

struct Random {
    bool use_fixed_seed;
    u32 main_seed;
    std::string qrng_path;
};

struct TestVectors {
    unsigned input_size;
    unsigned output_size;
    unsigned size_of_set;
    unsigned change_frequency;
};
} // namespace settings

namespace settings {
struct Circuit {
    unsigned size_of_layer;
    unsigned num_of_layers;

    struct FunctionSet {
        bool NOP;
        bool CONS;
        bool AND;
        bool NAND;
        bool OR;
        bool XOR;
        bool NOR;
        bool NOT;
        bool SHIL;
        bool SHIR;
        bool ROTL;
        bool ROTR;
        bool MASK;
    } function_set;
};
} // namespace settings
