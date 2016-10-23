// keccak.h
// 19-Nov-11  Markku-Juhani O. Saarinen <mjos@iki.fi>

#ifndef KECCAK_H
#define KECCAK_H

#include <inttypes.h>
#include <string.h>

#ifndef uint8_t
typedef unsigned char uint8_t;
#endif

#ifndef ROTL64
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))
#endif

class Keccak2 {
private:
    int rounds, r, c;
    unsigned char key[16];
    
public:
    Keccak2();
    virtual ~Keccak2();
        
    virtual unsigned getId() const { return 1; }
    virtual unsigned getInputBlockSize() const        { return 16; };
    virtual unsigned getOutputBlockSize() const       { return 16; };
    virtual unsigned getKeyBlockSize()  const         { return 16; };
    virtual int setNumRounds(int rounds)              { this->rounds = rounds; return 1; };
    virtual int getNumRounds() const                  { return this->rounds; }
        
    void set_parameters(int r, int c);
    
    // update the state with given number of rounds
    void keccakf(uint64_t st[25], int rounds) const;

    // compute a keccak hash (md) of given byte length from "in"
    int keccak(const uint8_t *in, int inlen, uint8_t *md, int mdlen) const;
    
    virtual int evaluate(const unsigned char * input, unsigned char * output) const;
    virtual int evaluate(const unsigned char *input, const unsigned char *key, unsigned char *output ) const;
    
    inline virtual int prepareKey(const unsigned char * key) 
    { memcpy(this->key, key, 16); return 1; }
    
    inline virtual int evaluateWithPreparedKey(const unsigned char * input, unsigned char * output) const
    { return evaluate(input, this->key, output); return 1;}
};
#endif

