#ifndef ENCRYPTORDECRYPTOR_H
#define ENCRYPTORDECRYPTOR_H

#include "EACglobals.h"
#include "EstreamInterface.h"
#include "EstreamConstants.h"

class EncryptorDecryptor {
public:
    /** array of used ciphers
      * - first index:
      *   0: use algorithm1 with its settings
      *   1: use algorithm2 with its settings
      * - second index:
      *   0: use stream for encrypting
      *   1: use stream for decrypting
      * - streams for encrypting and decrypting are separate to ensure the same internal state
      */
    EstreamInterface* m_ciphers[2][2];

    /** array of internal states if ciphers
      * (settings correspond to m_ciphers)
      */
    void* m_internalStates[2][2];

    //! array for key used
    unsigned char m_key[STREAM_BLOCK_SIZE];
    //! array for initialization vector used
    unsigned char m_iv[STREAM_BLOCK_SIZE];
    //! was initialization vector already set?
    bool m_setIV;
    //! was key already set?
    bool m_setKey;
public:
    /** constructor, allocates ciphers according to loaded settings
      */
    EncryptorDecryptor();

    /** destructor, release memory
      */
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
      * @param cipherNumber 0: use algorithm1 with its settings
      *                     1: use algorithm2 with its settings
      * @param streamNumber 0: use stream for encrypting
      *                     1: use stream for decrypting
      * @param length       plaintext length (if not set, testVectorLength is used)
      * @return status
      */
    int encrypt(unsigned char* plain, unsigned char* cipher, int cipherNumber, int streamNumber, int length = 0);

    /** decrypt given ciphertext
      * @param cipher       ciphertext pointer
      * @param plain        plaintext pointer
      * @param cipherNumber 0: use algorithm1 with its settings
      *                     1: use algorithm2 with its settings
      * @param streamNumber 0: use stream for encrypting
      *                     1: use stream for decrypting
      * @param length       plaintext length (if not set, testVectorLength is used)
      * @return status
      */
    int decrypt(unsigned char* cipher, unsigned char* plain, int cipherNumber, int streamNumber, int length = 0);
};

#endif
