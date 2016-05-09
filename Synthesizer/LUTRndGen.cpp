#include "LUTRndGen.h"

LUTRndGen::LUTRndGen(unsigned long seed, int m_LevelOfRandomness) {
	this->m_LevelOfRandomness = m_LevelOfRandomness;
	LUTSetRfunc(&S, seed, m_LevelOfRandomness);
}

void LUTRndGen::read(Dataset& data) {
	u8* dataPtr = data.data();
	int byteSize = data.num_of_tvs()*data.tv_size();
	int numIter = byteSize / 16; //LUT generates 16 bytesof rand data
	
	for (int i = 0; i < byteSize; i+= 16)
	{
		LUTUpdate(&S, 1);
		memcpy(dataPtr, S.stateBytes, 16);
	}

}



u64 LUTRndGen::operator()() {
	LUTUpdate(&S, 1);
	return S.state[0];
}





