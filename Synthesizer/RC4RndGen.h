#ifndef RC4_RNDGEN_H
#define RC4_RNDGEN_H

#include <string>
#include "arcfour.h"
#include "RandGen.h"
#include "../core/dataset.h"
#include "../core/project.h"

class RC4RndGen : public Stream{
public:
	RC4RndGen(unsigned long seed); //seed used as key 
	void read(Dataset& data);
private:
	u8 state[256]; 
};

#endif


