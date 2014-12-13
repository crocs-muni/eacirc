#ifndef OCB_H
#define OCB_H

#include "../../CaesarInterface.h"

class Ocb : public CaesarInterface {
    int m_mode;
    const int maxNumRounds = -1;
    const int maxMode = 9;
public:
    Ocb(int numRounds, int mode);
    ~Ocb();
    int encrypt(bits_t *c, length_t *clen, const bits_t *m, length_t mlen,
                        const bits_t *ad, length_t adlen, const bits_t *nsec, const bits_t *npub,
                        const bits_t *k);
    int decrypt(bits_t *m, length_t *outputmlen, bits_t *nsec,
                        const bits_t *c, length_t clen, const bits_t *ad, length_t adlen,
                        const bits_t *npub, const bits_t *k);
    std::string shortDescription() const;
};

#endif // OCB_H
