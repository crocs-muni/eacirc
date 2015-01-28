#ifndef ESTREAM_INTERFACE_H
#define ESTREAM_INTERFACE_H

#include "EstreamConstants.h"
#include "projects/estream/ciphers/ecrypt-portable.h"

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

#include "projects/estream/ciphers/abc/ecrypt-sync.h"
#include "projects/estream/ciphers/achterbahn/ecrypt-sync.h"
#include "projects/estream/ciphers/cryptmt/ecrypt-sync.h"
#include "projects/estream/ciphers/decim/ecrypt-sync.h"
#include "projects/estream/ciphers/dicing/ecrypt-sync.h"
#include "projects/estream/ciphers/dragon/ecrypt-sync.h"
#include "projects/estream/ciphers/edon80/ecrypt-sync.h"
#include "projects/estream/ciphers/ffcsr/ecrypt-sync.h"
#include "projects/estream/ciphers/fubuki/ecrypt-sync.h"
#include "projects/estream/ciphers/grain/ecrypt-sync.h"
#include "projects/estream/ciphers/hc-128/ecrypt-sync.h"
#include "projects/estream/ciphers/hermes/ecrypt-sync.h"
#include "projects/estream/ciphers/lex/ecrypt-sync.h"
#include "projects/estream/ciphers/mag/ecrypt-sync.h"
#include "projects/estream/ciphers/mickey/ecrypt-sync.h"
#include "projects/estream/ciphers/mir-1/ecrypt-sync.h"
#include "projects/estream/ciphers/pomaranch/ecrypt-sync.h"
#include "projects/estream/ciphers/py/ecrypt-sync.h"
#include "projects/estream/ciphers/rabbit/ecrypt-sync.h"
#include "projects/estream/ciphers/salsa20/ecrypt-sync.h"
#include "projects/estream/ciphers/sfinks/ecrypt-sync.h"
#include "projects/estream/ciphers/sosemanuk/ecrypt-sync.h"
#include "projects/estream/ciphers/tea/ecrypt-sync.h"
//#include "projects/estream/ciphers/trivium/ecrypt-sync.h"     // stopped working after IDE update
#include "projects/estream/ciphers/tsc-4/ecrypt-sync.h"
#include "projects/estream/ciphers/wg/ecrypt-sync.h"
//#include "projects/estream/ciphers/yamb/ecrypt-sync.h"        // stopped working after IDE update
#include "projects/estream/ciphers/zk-crypt/ecrypt-sync.h"

#endif
