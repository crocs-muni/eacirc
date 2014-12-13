#include "Ocb.h"
#include "Ocb_encrypt.h"
#include "../common/api.h"
#include "EACglobals.h"

Ocb::Ocb(int numRounds, int mode)
    : CaesarInterface(CAESAR_AESGCM, numRounds, CRYPTO_KEYBYTES, CRYPTO_NSECBYTES, CRYPTO_NPUBBYTES, CRYPTO_ABYTES), m_mode(mode) {
    if (numRounds < -1 || numRounds > maxNumRounds) {
        mainLogger.out(LOGGER_WARNING) << "Weird number of rouds (" << numRounds << ") for " << shortDescription() << endl;
    }
    if (numRounds == -1) {
        Ocb_raw::numRounds = maxNumRounds;
        CaesarCommon::numRounds = maxNumRounds;
    } else {
        Ocb_raw::numRounds = m_numRounds;
        CaesarCommon::numRounds = m_numRounds;
    }
    if (mode < 1 || mode > maxMode) {
        mainLogger.out(LOGGER_WARNING) << "Weird mode (" << mode << ") for " << shortDescription() << endl;
    }
    Ocb_raw::mode = mode;
}

Ocb::~Ocb() { }

int Ocb::encrypt(bits_t *c, length_t *clen, const bits_t *m, length_t mlen,
                       const bits_t *ad, length_t adlen, const bits_t *nsec, const bits_t *npub,
                       const bits_t *k) {
    return Ocb_raw::crypto_aead_encrypt(c, clen, m, mlen, ad, adlen, nsec, npub, k);
}

int Ocb::decrypt(bits_t *m, length_t *outputmlen, bits_t *nsec,
                       const bits_t *c, length_t clen, const bits_t *ad, length_t adlen,
                       const bits_t *npub, const bits_t *k) {
    return Ocb_raw::crypto_aead_decrypt(m, outputmlen, nsec, c, clen, ad, adlen, npub, k);
}

std::string Ocb::shortDescription() const {
    return string("OCB (mode ")+to_string(m_mode)+")";
}
