#include "LUT.h"
#include <string.h>
#include <random>
#include <iostream>
#include <vector>



void LUTSetRfunc(LUT_CTX *S, unsigned long seed , int LUT_HW){
	std::mt19937_64 gen(seed);
	std::uniform_int_distribution<unsigned int> dist(0, 16 * 256 * 128 - 1);

	u64 * luts = reinterpret_cast<u64 *>(S);

	/* compute number of bytes mod 64 */
	S->stateBytes = (u8*)S->state;

	/* LUTs are  cleared*/
	for (size_t i = 0; i < 16*256*2; i++)luts[i] = 0;
	
	/* HW_LUT of bits of LUT definning array is set to 1*/
	while (LUT_HW != 0) {
		const u64 bit = dist(gen); //which bit we want to set
		const u64 mask = u64(1) << (bit % 64);

		/* if bit is already set */
		if (luts[bit / 64] & mask)
			continue;

		/*otherwise set bit */
		luts[bit / 64] |= mask;
		--LUT_HW;
	}	
	/*internal state set to 0 */
	S->state[0] = S->state[1] = 0;
}

/*vector<int>& LUTSetBits(LUT_CTX *S){
	int  numBits = 16 * 256 * 128;
	int numBytes = 16 * 256 * 16;
	u8 byte;
	std::vector<int> setBits;
	setBits.reserve(numBits);


	for (int i = 0; i < numBytes; ++i) {
		byte = S->stateBytes[i];
		for (int j = 0; j < 8; ++j) {
			if(byte&1)setBits.push_back(i*8+i);
		}
	}
	return setBits;
}*/

void LUTLoadRfunc(LUT_CTX *S, u8* Rfunc){
	memcpy(S->stateBytes, Rfunc, 16*256*16);
}

void LUTSetState(LUT_CTX *S, u8* inn_state){
	memcpy(S->stateBytes,inn_state,16);
}
void LUTRound(LUT_CTX *S){
	u64 tmp_state[2] = { 0, 0 };
	u8 idx;
	/* XOR of LUT values (for 16 LUTs) corresponding to 16 Bytes of internal state  */
	for (int i = 0; i < 16; i++)
	{
		idx = S->stateBytes[i];
		tmp_state[0] ^= S->LUTs[i][idx][0];
		tmp_state[1] ^= S->LUTs[i][idx][1];
	}
	S->state[0] ^= tmp_state[0];
	S->state[1] ^= tmp_state[1];
}
void LUTUpdate(LUT_CTX *S, int num_round){
	for (size_t i = 0; i < num_round; i++)
		LUTRound(S);
}
