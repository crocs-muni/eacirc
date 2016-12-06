#pragma once

#include <memory>
#include "block_cipher.h"

#include <algorithm>
#include <core/memory.h>

#include "ciphers/tea/block-sync.h"


namespace block {
    //static std::unique_ptr<block_cipher> make_block_cipher(const std::string& algorithm, unsigned rounds);

    struct block_cipher;

    std::unique_ptr<block_cipher> make_block_cipher(const std::string& name, unsigned round) {
        // clang-format off
        if (name == "TEA")              return std::make_unique<tea>(round);
        // clang-format on

        throw std::runtime_error("requested block cipher named \"" + name +
                                 "\" is either broken or does not exists");
    }
};
