#include "LUT.h"
#include <string.h>
#include <random>
#include <iostream>

void LUTSetRfunc(LUT_CTX *S, u64* seed, int count){
	std::random_device rd;
	std::mt19937_64 gen1(seed[1]), gen2(seed[2]);
	std::uniform_int_distribution<u64> dist(0, 16 * 256 * 2 - 1);

	u64 * luts = reinterpret_cast<u64 *>(S);

	while (count != 0) {
		const u64 idx = dist(gen1) ^ dist(gen2);
		const u64 mask = u64(1) << (idx % 64);

		if (luts[idx / 64] & mask)
			continue;

		luts[idx / 64] |= mask;
		--count;
	}
}

void LUTSetRfunc(LUT_CTX *S, unsigned long seed , int count){
	std::random_device rd;
	std::mt19937_64 gen1(seed), gen2(seed+1);
	std::uniform_int_distribution<u64> dist(0, 16 * 256 * 128 - 1);

	u64 * luts = reinterpret_cast<u64 *>(S);

	S->stateBytes = (u8*)S->state;

	for (size_t i = 0; i < 16*256*2; i++)
	{
		luts[i] = 0;
	}

	while (count != 0) {
		const u64 idx = dist(gen1);
		const u64 mask = u64(1) << (idx % 64);

		//std::cout << idx << std::endl;
		if (luts[idx / 64] & mask)
			continue;

		luts[idx / 64] |= mask;
		--count;
		//std::cout << count << std::endl;
	}

	
}

void LUTLoadRfunc(LUT_CTX *S, u8* Rfunc){
	memcpy(S->stateBytes, Rfunc, 16*256*16);
}
void LUTSetState(LUT_CTX *S, u8* inn_state){
	memcpy(S->stateBytes,inn_state,16);
}
void LUTRound(LUT_CTX *S){
	u64 tmp_state[2] = { 0, 0 };
	u8 idx;
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
