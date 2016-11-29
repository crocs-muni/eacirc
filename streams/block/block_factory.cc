#include "block_factory.h"
#include "block_interface.h"
#include <algorithm>
#include <core/memory.h>

#include "ciphers/tea/block-sync.h"


std::unique_ptr<block_interface> block_factory::make_cipher(const std::string& name, unsigned round) {
    // clang-format off
    if (name == "TEA")              return std::make_unique<ECRYPT_TEA>(round);
    // clang-format on

    throw std::runtime_error("requested block cipher named \"" + name +
                             "\" is either broken or does not exists");
}
