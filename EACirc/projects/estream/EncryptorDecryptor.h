#ifndef ENCRYPTORDECRYPTOR_H
#define ENCRYPTORDECRYPTOR_H

#include "EACglobals.h"
#include "EstreamInterface.h"
#include "EstreamConstants.h"

class EncryptorDecryptor {
public:
	EstreamInterface* ecryptarr[4];
	void* ctxarr[4];
    unsigned char m_key[STREAM_BLOCK_SIZE];
    unsigned char m_iv[STREAM_BLOCK_SIZE];
    //! was initialization vector already set?
    bool m_setIV;
    //! was key already set?
    bool m_setKey;
public:
    EncryptorDecryptor();
    ~EncryptorDecryptor();

    /** set initialization vector for current cipher according to loaded settings
      * @return status
      */
    int setupIV();

    /** set key for current cipher according to loaded settings
      * @return status
      */
    int setupKey();

    /** encrypt given plaintext
      * @param plain        plaintext pointer
      * @param cipher       ciphertext pointer
      * @param streamnum    ??? TBD
      * @param length       plaintext length (if not set, testVectorLength is used)
      * @return status
      */
    int encrypt(unsigned char* plain, unsigned char* cipher, int streamnum = 0, int length = 0);

    /** decrypt given ciphertext
      * @param cipher       ciphertext pointer
      * @param plain        plaintext pointer
      * @param streamnum    ??? TBD
      * @param length       plaintext length (if not set, testVectorLength is used)
      * @return status
      */
    int decrypt(unsigned char* cipher, unsigned char* plain, int streamnum = 0, int length = 0);
};

#endif
