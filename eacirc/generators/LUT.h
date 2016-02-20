
#ifndef LUT_H
#define LUT_H

#define LUT_STATE_LENGTH           16  

/* typedef a 32 bit type */
typedef unsigned long long u64;
typedef unsigned char u8;


/* Data structure for LUT  */
typedef struct
{
	u64 LUTs[16][256][2]; // 16x LUTs each LUT: 8(bits)->128(bits)
	u64 state[2];		  //internal state 128 bits
	u8* stateBytes;		  //internal state as Bytes (stateBytes==state)
} LUT_CTX;


void LUTSetRfunc(LUT_CTX *S, unsigned long seed = 0, int count = 262144); //round function setting
void LUTLoadRfunc(LUT_CTX *S, u8* Rfunc); //load round functionfrom array
void LUTSetState(LUT_CTX *S, u8* internal_state); //set internal state
void LUTRound(LUT_CTX *S); //perform round 
void LUTUpdate(LUT_CTX *S, int num_round = 1); //Update internal state

#endif  //LUT_H
