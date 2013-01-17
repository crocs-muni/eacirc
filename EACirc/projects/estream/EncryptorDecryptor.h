#ifndef ENCRYPTORDECRYPTOR_H
#define ENCRYPTORDECRYPTOR_H

#include "EACglobals.h"
#include "estreamInterface.h"

class EncryptorDecryptor {
	EstreamInterface* ecryptarr[4];
	void* ctxarr[4];
	unsigned char key[STREAM_BLOCK_SIZE];
	unsigned char iv[STREAM_BLOCK_SIZE];
public:
    EncryptorDecryptor();
    ~EncryptorDecryptor();
    void encrypt(unsigned char* plain, unsigned char* cipher, int streamnum = 0, int length = 0);
    void decrypt(unsigned char* cipher, unsigned char* plain, int streamnum = 0, int length = 0);
};

#endif
