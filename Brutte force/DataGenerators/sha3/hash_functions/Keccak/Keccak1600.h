/* 
 * File:   KeccakFull.h
 * Author: ph4r05
 *
 * Created on June 19, 2014, 2:23 PM
 */

#ifndef KECCAKFULL_H
#define	KECCAKFULL_H


#include <inttypes.h>
#include <string.h>
#include "base.h"
#include "ICipher.h"

#ifndef uint8_t
typedef unsigned char uint8_t;
#endif

#ifndef ROTL64
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))
#endif

class Keccak1600 {
private:
    int rounds;
    
public:
    Keccak1600();
    virtual ~Keccak1600();
        
    virtual unsigned getId() const { return 2; }
    virtual unsigned getInputBlockSize() const        { return 128; };
    virtual unsigned getOutputBlockSize() const       { return 128; };
    virtual unsigned getKeyBlockSize()  const         { return 0; };
    virtual int setNumRounds(int rounds);
    virtual int getNumRounds() const                  { return this->rounds; }
        
    virtual int evaluate(const unsigned char * input, unsigned char * output) const;
    virtual int evaluate(const unsigned char *input, const unsigned char *key, unsigned char *output ) const;

};

#endif	/* KECCAKFULL_H */

