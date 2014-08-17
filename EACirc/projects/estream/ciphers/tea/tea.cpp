
#include <iostream>
#include "ecrypt-sync.h"

static const u32 delta = 0x9e3779b9;

void ECRYPT_TEA::ECRYPT_init() { }

void ECRYPT_TEA::ECRYPT_keysetup(void *ctx, const u8 *key, u32 keysize, u32 ivsize)
{
    TEA_ctx* teaCtx = (TEA_ctx*) ctx;
    //copy key of u8 to array of u32
    for(int i = 0; i < 4; i++) teaCtx->key[i] = U8TO32_LITTLE(key + 4*i);
    for(int j = 0; j < 2; j++) teaCtx->input[j] = 0;
}

void ECRYPT_TEA::ECRYPT_ivsetup(void *ctx, const u8 *iv) { }

void ECRYPT_TEA::ECRYPT_encrypt_bytes(void *ctx, const u8 *plaintext, u8 *ciphertext, u32 msglen)
{
    TEA_ctx* teaCtx = (TEA_ctx*) ctx;
    if(!msglen) return;
    for(unsigned int i = 0; i < msglen + 8; i += 8) {
        //copy plaintext of u8 to array of u32
        for(int j = 0; j < 2; j++) teaCtx->input[j] = U8TO32_LITTLE(plaintext + i + 4*j);

        u32 sum = 0;
        for (int j = 0; j < numRounds; j++) {
            sum += delta;
            teaCtx->input[0] += ((teaCtx->input[1]<<4) + teaCtx->key[0])
                    ^ (teaCtx->input[1] + sum)
                    ^ ((teaCtx->input[1]>>5) + teaCtx->key[1]);

            teaCtx->input[1] += ((teaCtx->input[0]<<4) + teaCtx->key[2])
                    ^ (teaCtx->input[0] + sum)
                    ^ ((teaCtx->input[0]>>5) + teaCtx->key[3]);
        }
        //debug
        u32 tmp0 = teaCtx->input[0];
        u32 tmp1 = teaCtx->input[1];
        for(int j = 0; j < 2; j++) U32TO8_LITTLE(ciphertext + i + 4*j, teaCtx->input[j]);
    }
}

void ECRYPT_TEA::ECRYPT_decrypt_bytes(void *ctx, const u8 *ciphertext, u8 *plaintext, u32 msglen)
{
    TEA_ctx* teaCtx = (TEA_ctx*) ctx;
    if(!msglen) return;
    for(unsigned int i = 0; i < msglen + 8; i += 8) {
        //copy plaintext of u8 to array of u32
        for(int j = 0; j < 2; j++) teaCtx->input[j] = U8TO32_LITTLE(ciphertext + i + 4*j);

        u32 sum = delta * numRounds;

        for (int j = 0; j < numRounds; j++) {
            teaCtx->input[1] -= ((teaCtx->input[0]<<4) + teaCtx->key[2])
                    ^ (teaCtx->input[0] + sum)
                    ^ ((teaCtx->input[0]>>5) + teaCtx->key[3]);

            teaCtx->input[0] -= ((teaCtx->input[1]<<4) + teaCtx->key[0])
                    ^ (teaCtx->input[1] + sum)
                    ^ ((teaCtx->input[1]>>5) + teaCtx->key[1]);
            sum -= delta;
        }
        //debug
        u32 tmp0 = teaCtx->input[0];
        u32 tmp1 = teaCtx->input[1];
        for(int j = 0; j < 2; j++) U32TO8_LITTLE(plaintext + i + 4*j, teaCtx->input[j]);
    }
}
