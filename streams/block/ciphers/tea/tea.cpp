#include "block-sync.h"
#include <iostream>

namespace block {

    std::uint32_t u8_to_u32_copy(const uint8_t* in) {
        return std::uint32_t((in[0] << 24) + (in[1] << 16) + (in[2] << 8) + in[3]);
    }

    void u32_to_u8_copy(uint8_t* out, const uint32_t in) {
        out[0] = uint8_t(in >> 24);
        out[1] = uint8_t(in >> 16);
        out[2] = uint8_t(in >> 8);
        out[3] = uint8_t(in);
    }

    void tea::keysetup(const std::uint8_t* key, std::uint32_t keysize, std::uint32_t ivsize) {
        if (keysize != 4)
            throw std::runtime_error("tea keysize should be 4 B");
        if (ivsize != 4)
            throw std::runtime_error("tea ivsize should be 4 B");

        tea_ctx* teaCtx = &_ctx;
        // copy key of std::uint8_t to array of std::uint32_t
        for (int i = 0; i < 4; i++)
            teaCtx->key[i] = u8_to_u32_copy(key + 4 * i);
        for (int j = 0; j < 2; j++)
            teaCtx->input[j] = 0;
    }

    void tea::ivsetup(const std::uint8_t* iv) {
        throw std::runtime_error("not implemented yet");
    }

    void
    tea::encrypt(const std::uint8_t* plaintext, std::uint8_t* ciphertext, std::uint32_t msglen) {
        tea_ctx* teaCtx = &_ctx;
        if (!msglen)
            return;
        for (unsigned int i = 0; i < msglen; i += 8) {
            // copy plaintext of std::uint8_t to array of std::uint32_t
            for (int j = 0; j < 2; j++)
                teaCtx->input[j] = u8_to_u32_copy(plaintext + i + 4 * j);

            std::uint32_t sum = 0;
            for (unsigned j = 0; j < _rounds; j++) {
                sum += delta;
                teaCtx->input[0] += ((teaCtx->input[1] << 4) + teaCtx->key[0]) ^
                                    (teaCtx->input[1] + sum) ^
                                    ((teaCtx->input[1] >> 5) + teaCtx->key[1]);

                teaCtx->input[1] += ((teaCtx->input[0] << 4) + teaCtx->key[2]) ^
                                    (teaCtx->input[0] + sum) ^
                                    ((teaCtx->input[0] >> 5) + teaCtx->key[3]);
            }
            for (int j = 0; j < 2; j++)
                u32_to_u8_copy(ciphertext + i + 4 * j, teaCtx->input[j]);
        }
    }

    void
    tea::decrypt(const std::uint8_t* ciphertext, std::uint8_t* plaintext, std::uint32_t msglen) {
        tea_ctx* teaCtx = &_ctx;
        if (!msglen)
            return;
        for (unsigned int i = 0; i < msglen; i += 8) {
            // copy plaintext of std::uint8_t to array of std::uint32_t
            for (int j = 0; j < 2; j++)
                teaCtx->input[j] = u8_to_u32_copy(ciphertext + i + 4 * j);

            std::uint32_t sum = delta * _rounds;

            for (int j = 0; j < _rounds; j++) {
                teaCtx->input[1] -= ((teaCtx->input[0] << 4) + teaCtx->key[2]) ^
                                    (teaCtx->input[0] + sum) ^
                                    ((teaCtx->input[0] >> 5) + teaCtx->key[3]);

                teaCtx->input[0] -= ((teaCtx->input[1] << 4) + teaCtx->key[0]) ^
                                    (teaCtx->input[1] + sum) ^
                                    ((teaCtx->input[1] >> 5) + teaCtx->key[1]);
                sum -= delta;
            }
            for (int j = 0; j < 2; j++)
                u32_to_u8_copy(plaintext + i + 4 * j, teaCtx->input[j]);
        }
    }
}
