#include <iostream>
#include "../core/base.h"
#include "../Synthesizer/LUTRndGen.h"
#include "../Synthesizer/PatternRndGen.h"
#include "../Synthesizer/MersenneTwisterRndGen.h"
#include "../Synthesizer/QuantumGen.h"
#include "../Synthesizer/Synthesizer.h"
#include "arcfour.h"

using namespace std;

int main() {

	
	Setting S;
	std::vector<std::vector<short>> terms;
	terms.resize(2);
	terms[0].push_back(1);
	terms[0].push_back(2);
	terms[1].push_back(3);
	terms[1].push_back(4);

	S.filename = "test.bin";
	S.genID = 0;
	S.levelOfRandomness = 1000;
	S.seed = 0;
	S.terms = terms;
	S.tv_size = 8;


	Synthesizer syn(S);
	Dataset data(16, 1);
	//LUTRndGen l(200000,0);
	//l.read(data);

	//PatternGen l(terms,64,1);
	//MersenneTwisterRndGen l(1);
	//QuantumRndGen l(1,"test.bin");

	
	for (size_t i = 0; i < 10; i++)
	{
		syn.read(data);
		for (int i = 0; i < 8; ++i) {
			u8 tmp = data.data()[i];
			for (int j = 0; j < 8; ++j) {
				std::cout << (tmp & 1) ? 1 : 0;
				tmp >>= 1;
			}
			//std::cout << " ";
		}
		std::cout << std::endl;
	}

	return 0;
}