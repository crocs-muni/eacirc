#ifndef ESTREAM_INTERFACE_H
#define ESTREAM_INTERFACE_H

#include "ciphers/ecrypt-portable.h"

typedef void* ECRYPT_ctx;

class EstreamInterface {
    public:
        int numRounds;
        EstreamInterface() : numRounds(0) {}
        virtual ~EstreamInterface() {}
        virtual void ECRYPT_init(void) = 0;
        virtual void ECRYPT_keysetup(void* ctx, const u8 * key, u32 keysize,	u32 ivsize) = 0;
        virtual void ECRYPT_ivsetup(void* ctx, const u8 * iv) = 0;
        virtual void ECRYPT_encrypt_bytes(void* ctx, const u8 * plaintext, u8 * ciphertext, u32 msglen) = 0;
        virtual void ECRYPT_decrypt_bytes(void* ctx, const u8 * ciphertext, u8 * plaintext, u32 msglen) = 0;
};

//#include "ciphers/abc/ecrypt-sync.h"
//#include "ciphers/achterbahn/ecrypt-sync.h"
//#include "ciphers/cryptmt/ecrypt-sync.h"
#include "ciphers/decim/ecrypt-sync.h"
//#include "ciphers/dicing/ecrypt-sync.h"
//#include "ciphers/dragon/ecrypt-sync.h"
//#include "ciphers/edon80/ecrypt-sync.h"
//#include "ciphers/ffcsr/ecrypt-sync.h"
//#include "ciphers/fubuki/ecrypt-sync.h"
//#include "ciphers/grain/ecrypt-sync.h"
//#include "ciphers/hc-128/ecrypt-sync.h"
//#include "ciphers/hermes/ecrypt-sync.h"
//#include "ciphers/lex/ecrypt-sync.h"
//#include "ciphers/mag/ecrypt-sync.h"
//#include "ciphers/mickey/ecrypt-sync.h"
//#include "ciphers/mir-1/ecrypt-sync.h"
//#include "ciphers/pomaranch/ecrypt-sync.h"
//#include "ciphers/py/ecrypt-sync.h"
//#include "ciphers/rabbit/ecrypt-sync.h"
//#include "ciphers/salsa20/ecrypt-sync.h"
//#include "ciphers/sfinks/ecrypt-sync.h"
//#include "ciphers/sosemanuk/ecrypt-sync.h"
#include "ciphers/tea/ecrypt-sync.h"
////#include "ciphers/trivium/ecrypt-sync.h"     // stopped working after IDE update
//#include "ciphers/tsc-4/ecrypt-sync.h"
//#include "ciphers/wg/ecrypt-sync.h"
////#include "ciphers/yamb/ecrypt-sync.h"        // stopped working after IDE update
//#include "ciphers/zk-crypt/ecrypt-sync.h"

#endif
