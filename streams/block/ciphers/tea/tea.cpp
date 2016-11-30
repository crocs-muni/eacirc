
#include "block-sync.h"
#include <iostream>

static const std::uint32_t delta = 0x9e3779b9;

void tea::keysetup(const std::uint8_t* key, std::uint32_t keysize, std::uint32_t ivsize) {
    TEA_ctx* teaCtx = &_ctx;
    // copy key of std::uint8_t to array of std::uint32_t
    for (int i = 0; i < 4; i++)
        teaCtx->key[i] = std::uint8_tTO32_LITTLE(key + 4 * i);
    for (int j = 0; j < 2; j++)
        teaCtx->input[j] = 0;
}

void tea::ivsetup(const std::uint8_t* iv) {}

void tea::encrypt(const std::uint8_t* plaintext, std::uint8_t* ciphertext, std::uint32_t msglen) {
    TEA_ctx* teaCtx = &_ctx;
    if (!msglen)
        return;
    for (unsigned int i = 0; i < msglen; i += 8) {
        // copy plaintext of std::uint8_t to array of std::uint32_t
        for (int j = 0; j < 2; j++)
            teaCtx->input[j] = (*std::uint32_t) (plaintext + i + 4 * j);

        std::uint32_t sum = 0;
        for (int j = 0; j < _rounds; j++) {
            sum += delta;
            teaCtx->input[0] += ((teaCtx->input[1] << 4) + teaCtx->key[0]) ^
                                (teaCtx->input[1] + sum) ^
                                ((teaCtx->input[1] >> 5) + teaCtx->key[1]);

            teaCtx->input[1] += ((teaCtx->input[0] << 4) + teaCtx->key[2]) ^
                                (teaCtx->input[0] + sum) ^
                                ((teaCtx->input[0] >> 5) + teaCtx->key[3]);
        }
        for (int j = 0; j < 2; j++)
            std::uint32_tTO8_LITTLE(ciphertext + i + 4 * j, teaCtx->input[j]);
    }
}

void tea::decrypt(const std::uint8_t* ciphertext, std::uint8_t* plaintext, std::uint32_t msglen) {
    TEA_ctx* teaCtx = &_ctx;
    if (!msglen)
        return;
    for (unsigned int i = 0; i < msglen; i += 8) {
        // copy plaintext of std::uint8_t to array of std::uint32_t
        for (int j = 0; j < 2; j++)
            teaCtx->input[j] = std::uint8_tTO32_LITTLE(ciphertext + i + 4 * j);

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
            std::uint32_tTO8_LITTLE(plaintext + i + 4 * j, teaCtx->input[j]);
    }
}
