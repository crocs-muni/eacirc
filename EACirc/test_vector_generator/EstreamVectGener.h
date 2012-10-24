#ifndef TEST_VECT_GENER_ESTREAM_H
#define TEST_VECT_GENER_ESTREAM_H

#include "EACglobals.h"
#include "test_vector_generator/ITestVectGener.h"

class EstreamTestVectGener: public ITestVectGener {
	//private:
		unsigned char outputs[MAX_OUTPUTS];
		unsigned char inputs[MAX_INPUTS];
		int *numstats;
	public:
		EstreamTestVectGener();
        EstreamTestVectGener(const EstreamTestVectGener&) = delete;
        const EstreamTestVectGener& operator =(const EstreamTestVectGener&) = delete;
		void getTestVector();
		void generateTestVectors();
		void Init();
};

#endif
