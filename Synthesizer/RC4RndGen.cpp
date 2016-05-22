#include "RC4RndGen.h"

//seed used as key key
RC4RndGen::RC4RndGen(unsigned long seed) {
	arcfour_key_setup(state, (u8*)&seed, 4);
}

void RC4RndGen::read(Dataset& data) {
	u8* dataPtr = data.data();
	int byteSize = data.num_of_tvs()*data.tv_size();
	int numIter = byteSize / 256; //RC4 generates 256 bytes of rand data
	
	for (int i = 0; i < numIter; i++)
	{
		arcfour_generate_stream(state, dataPtr, 256);
		dataPtr += 256;
	}
	if( (byteSize % 256) != 0 )arcfour_generate_stream(state, dataPtr, byteSize % 256);

}






