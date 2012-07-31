#ifndef TEST_VECT_GENER_ESTREAM_H
#define TEST_VECT_GENER_ESTREAM_H

#include "ITestVectGener.h"

class EstreamTestVectGener: public ITestVectGener {
	//private:
		unsigned char outputs[MAX_OUTPUTS];
		unsigned char inputs[MAX_INPUTS];
		int *numstats;
	public:
		EstreamTestVectGener();
		void getTestVector();
		void generateTestVectors();
		void Init();
};

#endif