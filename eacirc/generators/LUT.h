
#ifndef LUT_H
#define LUT_H

#define LUT_STATE_LENGTH           16  

/* typedef a 32 bit type */
typedef unsigned long long u64;
typedef unsigned char u8;


/* Data structure for LUT  */
typedef struct
{
	u64 LUTs[16][256][2];
	u64 state[2];
	u8* stateBytes;
} LUT_CTX;


void LUTSetRfunc(LUT_CTX *S, u64* seed = 0, int count = 262144);
void LUTSetRfunc(LUT_CTX *S, unsigned long seed = 0, int count = 262144);
void LUTLoadRfunc(LUT_CTX *S, u8* Rfunc);
void LUTSetState(LUT_CTX *S, u8* inn_state);
void LUTRound(LUT_CTX *S);
void LUTUpdate(LUT_CTX *S, int num_round = 1);



#endif  //LUT_H
