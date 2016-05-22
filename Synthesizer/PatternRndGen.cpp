#include"PatternRndGen.h"


PatternRndGen::PatternRndGen(std::vector<std::vector<short>>& terms, int testVectorSizeBits, unsigned long seed){
	gen.seed(seed);
	testVectorSize = (testVectorSizeBits + 63) / 64;
    _testVector = new u64(testVectorSize);
    _terms = terms;
}


void PatternRndGen::setRandomData() {
	for (int i = 0; i < testVectorSize; ++i) {
		_testVector[i] = gen();
	}
}

// polynomial is evaluated to zero - at least one variable in term is set to zero
void PatternRndGen::setPolynomial(){
    int variableToZero;

    for (int i = 0; i < _terms.size(); ++i) {
        //take rand variable of the term and set it to zero
        variableToZero = _terms[i][gen() % _terms[i].size()];
        _testVector[variableToZero / 64] &= ~(1 << (variableToZero % 64));
    }
}



void PatternRndGen::read(Dataset& data) {
	u8* dataPtr = data.data();
	int numIter = data.num_of_tvs(), tvSize = data.tv_size();

	for (int i = 0; i < numIter; i++)
	{
		setRandomData();
		setPolynomial();
		memcpy(dataPtr,(void*)&_testVector[0], tvSize);
		dataPtr += tvSize;
	}

}




