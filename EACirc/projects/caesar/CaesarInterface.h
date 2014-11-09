#ifndef CAESARINTERFACE_H
#define CAESARINTERFACE_H

#include <string>
#include "CaesarConstants.h"

class CaesarInterface {
protected:
    //! CAESAR algorithm used
    int m_algorithm;
    //! number of rounds
    int m_numRounds;
    //! algorothm constants
    int m_keyLength;
    int m_secretMessageNumberLength;
    int m_publicMessageNumberLength;
    int m_cipertextOverhead;

public:
    CaesarInterface(int a, int nr, int kl, int smnl, int pmnl, int co);
    virtual ~CaesarInterface();

    /** encryption and decryption functions
     * c - ciphertext, clen - ciphertext length,
     * m - message, mlen - message length,
     * ad - associated data, adlen - associated data length,
     * nsec - secret message number (secret nonce),
     * npub - public message number (public nonce),
     * k -key
     */
    virtual int encrypt(bits_t *c, length_t *clen, const bits_t *m, length_t mlen,
                        const bits_t *ad, length_t adlen, const bits_t *nsec, const bits_t *npub,
                        const bits_t *k) = 0;
    virtual int decrypt(bits_t *m, length_t *outputmlen, bits_t *nsec,
                        const bits_t *c, length_t clen, const bits_t *ad, length_t adlen,
                        const bits_t *npub, const bits_t *k) = 0;

    /** get algorothm constatns */
    int getKeyLength();
    int getSecretMessageNumberLength();
    int getPublicMessageNumberLength();
    int getCipertextOverhead();

    /** human readable ag=lgorithm description */
    virtual std::string shortDescription() const = 0;

    /** allocate new caesar algorithm according to parameters
      * @param algorithm        CAESAR algorithm constant
      * @param numRounds        number of rounds used
      * @return allocated CAESAR algorithm obejct
      */
    static CaesarInterface* getCaesarFunction(int algorithm, int numRounds);
};

#endif // CAESARINTERFACE_H
