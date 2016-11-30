#pragma once

#include "../../block_cipher.h"


class ECRYPT_TEA : public block_cipher {

    /* Data structures */


    struct TEA_ctx {

        TEA_ctx() : input{0}, key{0} {}

        std::uint32_t input[2];
        std::uint32_t key[4];
    } _ctx;


public:
    ECRYPT_TEA(unsigned rounds)
        : block_cipher(rounds) {}


    void keysetup(const std::uint8_t* key,
                         std::uint32_t keysize,
                         std::uint32_t ivsize) override;


    void ivsetup(const std::uint8_t* iv) override;

    void encrypt(const std::uint8_t* plaintext,
                              std::uint8_t* ciphertext,
                              std::uint32_t msglen) override;

    void decrypt(const std::uint8_t* ciphertext,
                              std::uint8_t* plaintext,
                              std::uint32_t msglen) override;

};

