#pragma once

#include <memory>
#include "block_cipher.h"

#include <algorithm>
#include <core/memory.h>

#include "ciphers/tea/tea.h"
#include "ciphers/aes/aes.h"
#include "ciphers/rc4/rc4.h"


namespace block {
    struct block_cipher;

    std::unique_ptr<block_cipher> make_block_cipher(const std::string& name, unsigned round) {
        // clang-format off
        if (name == "TEA")  return std::make_unique<tea>(round);
        if (name == "AES")  return std::make_unique<aes>(round);
        if (name == "RC4")  return std::make_unique<rc4>(round, 16); // TODO: optional block_size
        // clang-format on

        throw std::runtime_error("requested block cipher named \"" + name +
                                 "\" is either broken or does not exists");
    }
};
