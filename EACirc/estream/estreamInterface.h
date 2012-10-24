#ifndef ESTREAM_INTERFACE_H
#define ESTREAM_INTERFACE_H

#include "estream/ciphers/ecrypt-portable.h"

typedef void* ECRYPT_ctx;

class EstreamInterface {
	public:
        int numRounds = 0;
        EstreamInterface() {}
        virtual ~EstreamInterface() {}
		virtual void ECRYPT_init(void) = 0;
		virtual void ECRYPT_keysetup(void* ctx, const u8 * key, u32 keysize,	u32 ivsize) = 0;
		virtual void ECRYPT_ivsetup(void* ctx, const u8 * iv) = 0;
		virtual void ECRYPT_encrypt_bytes(void* ctx, const u8 * plaintext, u8 * ciphertext, u32 msglen) = 0;
		virtual void ECRYPT_decrypt_bytes(void* ctx, const u8 * ciphertext, u8 * plaintext, u32 msglen) = 0;
};

#include "estream/ciphers/abc/ecrypt-sync.h"
#include "estream/ciphers/achterbahn/ecrypt-sync.h"
#include "estream/ciphers/cryptmt/ecrypt-sync.h"
#include "estream/ciphers/decim/ecrypt-sync.h"
#include "estream/ciphers/dicing/ecrypt-sync.h"
#include "estream/ciphers/dragon/ecrypt-sync.h"
#include "estream/ciphers/edon80/ecrypt-sync.h"
#include "estream/ciphers/ffcsr/ecrypt-sync.h"
#include "estream/ciphers/fubuki/ecrypt-sync.h"
#include "estream/ciphers/grain/ecrypt-sync.h"
#include "estream/ciphers/hc-128/ecrypt-sync.h"
#include "estream/ciphers/hermes/ecrypt-sync.h"
#include "estream/ciphers/lex/ecrypt-sync.h"
#include "estream/ciphers/mag/ecrypt-sync.h"
#include "estream/ciphers/mickey/ecrypt-sync.h"
#include "estream/ciphers/mir-1/ecrypt-sync.h"
#include "estream/ciphers/pomaranch/ecrypt-sync.h"
#include "estream/ciphers/py/ecrypt-sync.h"
#include "estream/ciphers/rabbit/ecrypt-sync.h"
#include "estream/ciphers/salsa20/ecrypt-sync.h"
#include "estream/ciphers/sfinks/ecrypt-sync.h"
#include "estream/ciphers/sosemanuk/ecrypt-sync.h"
#include "estream/ciphers/trivium/ecrypt-sync.h"
#include "estream/ciphers/tsc-4/ecrypt-sync.h"
#include "estream/ciphers/wg/ecrypt-sync.h"
#include "estream/ciphers/yamb/ecrypt-sync.h"
#include "estream/ciphers/zk-crypt/ecrypt-sync.h"

//constant for TEST_VECTOR_GENERATION_METHOD settings
#define ESTREAM_CONST 666

//ESTREAM TESTVECT GENERATION METHOD:
#define TESTVECT_ESTREAM_DISTINCT 667
#define TESTVECT_ESTREAM_PREDICT_KEY 668
#define TESTVECT_ESTREAM_BITS_TO_CHANGE 669

#define ESTREAM_GENTYPE_ZEROS 0
#define ESTREAM_GENTYPE_ONES 1
#define ESTREAM_GENTYPE_RANDOM 2
#define ESTREAM_GENTYPE_BIASRANDOM 3

// eStream cipher constants
const char* estreamToString(int cipher);
#define ESTREAM_ABC 1
#define ESTREAM_ACHTERBAHN 2
#define ESTREAM_CRYPTMT 3
#define ESTREAM_DECIM 4
#define ESTREAM_DICING 5
#define ESTREAM_DRAGON 6
#define ESTREAM_EDON80 7
#define ESTREAM_FFCSR 8
#define ESTREAM_FUBUKI 9
#define ESTREAM_GRAIN 10
#define ESTREAM_HC128 11
#define ESTREAM_HERMES 12
#define ESTREAM_LEX 13
#define ESTREAM_MAG 14
#define ESTREAM_MICKEY 15
#define ESTREAM_MIR1 16
#define ESTREAM_POMARANCH 17
#define ESTREAM_PY 18
#define ESTREAM_RABBIT 19
#define ESTREAM_SALSA20 20
#define ESTREAM_SFINKS 21
#define ESTREAM_SOSEMANUK 22
#define ESTREAM_TRIVIUM 23
#define ESTREAM_TSC4 24
#define ESTREAM_WG 25
#define ESTREAM_YAMB 26
#define ESTREAM_ZKCRYPT 27
#define ESTREAM_RANDOM 99

#endif
