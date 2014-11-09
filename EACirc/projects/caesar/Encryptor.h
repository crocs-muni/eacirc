#ifndef ENCRYPTOR_H
#define ENCRYPTOR_H

#include "CaesarConstants.h"
#include "CaesarInterface.h"

class Encryptor {
private:
    //! cipher to use
    CaesarInterface* m_cipher;
    //! key buffer
    bits_t* m_key;
    //! associated data
    bits_t* m_ad;
    //! secret message number
    bits_t* m_smn;
    //! public message number
    bits_t* m_pmn;
    //! buffer for decrypted message vor verification
    bits_t* m_decryptedMessage;
    //! buffer for decrypted secret message number
    bits_t* m_decryptedSmn;
    //! decrypted plaintext message length
    length_t m_decryptedMessageLength;
    //! setup already performed?
    bool m_setup;

public:
    /** constructor
     * - allocate cipher
     * - allocate buffers for key, secret message number
     *   public message number, plaintext, ciphertext
     */
    Encryptor();

    /** destructor, free buffers */
    ~Encryptor();

    /** setup key, smn, pmn
     * @return status
     */
    int setup();

    /** encrypt message and verify ciphertext
     * @param m         message to encrypt
     * @param mlen      length of the plaintext
     * @param c         ciphertext
     * @param clen      length of the ciphertext
     * @return          verification status
     */
    int encrypt(const bits_t *m, bits_t *c, length_t *clen);
};

#endif // ENCRYPTOR_H
