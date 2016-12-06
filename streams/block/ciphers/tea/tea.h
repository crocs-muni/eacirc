#pragma once

/**
 * Source: https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
 */

#include "../../block_cipher.h"

namespace block {

    class tea : public block_cipher {

        /* Data structures */

        struct tea_ctx {

            tea_ctx()
                : key{0} {}

            std::uint32_t key[4];
        } _ctx;

        const std::uint32_t _delta = 0x9e3779b9;
        const std::uint32_t _msglen = 64;

    public:
        tea(unsigned rounds)
            : block_cipher(rounds) {}

        void keysetup(const std::uint8_t* key, std::uint32_t keysize) override;

        void ivsetup(const std::uint8_t* iv, std::uint32_t ivsize) override;

        void encrypt(const std::uint8_t* plaintext,
                     std::uint8_t* ciphertext) override;

        void decrypt(const std::uint8_t* ciphertext,
                     std::uint8_t* plaintext) override;
    };
}
