#ifndef SYNTHESIZER_H
#define SYNTHESIZER_H

#include <random>
#include <string>
#include "../core/base.h"
#include "../core/project.h"
#include "LUTRndGen.h"
#include "QuantumGen.h"
#include "MersenneTwisterRndGen.h"
#include "PatternRndGen.h"
#include "RC4RndGen.h"

#include <iostream>
#include <stdlib.h>

struct Setting {
	int genID;
	u64 seed;
	int levelOfRandomness;
	std::vector<std::vector<short>> terms;
	int tv_size;
	std::string filename;
};

class Synthesizer{

public:
	Synthesizer(Setting S){
		if (S.genID == 0) gen = new MersenneTwisterRndGen(S.seed);
		if (S.genID == 1) gen = new QuantumRndGen(S.seed, S.filename);
		if (S.genID == 2) gen = new LUTRndGen(S.seed, S.levelOfRandomness);
		if (S.genID == 3) gen = new PatternRndGen(S.terms,S.tv_size,S.seed);
		if (S.genID == 4) gen = new RC4RndGen(S.seed);
	}
	
	void read(Dataset& data) {
		gen->read(data);
	}

private:
	Stream* gen;
};


#endif //SYNTHESIZER_H


