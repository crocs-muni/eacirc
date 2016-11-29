#pragma once

#include <memory>
#include "block_interface.h"


struct block_factory {
    static std::unique_ptr<block_interface> make_cipher(const std::string& algorithm, unsigned rounds);
};
