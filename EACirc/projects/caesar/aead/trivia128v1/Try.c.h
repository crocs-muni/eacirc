#ifndef TRY.C_H
#define TRY.C_H

#include "../../CaesarInterface.h"

class Try.c : public CaesarInterface {
    const int maxNumRounds = -1;
public:
    Try.c(int numRounds);
    ~Try.c();
    int encrypt(bits_t *c, length_t *clen, const bits_t *m, length_t mlen,
                        const bits_t *ad, length_t adlen, const bits_t *nsec, const bits_t *npub,
                        const bits_t *k);
    int decrypt(bits_t *m, length_t *outputmlen, bits_t *nsec,
                        const bits_t *c, length_t clen, const bits_t *ad, length_t adlen,
                        const bits_t *npub, const bits_t *k);
    std::string shortDescription() const;
};

#endif // TRY.C_H
