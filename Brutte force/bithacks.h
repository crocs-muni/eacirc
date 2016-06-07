//
// Created by syso on 19. 5. 2016.
//
//https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetNaive

#ifndef DYNAMIC_BITSET_BITHACKS_H
#define DYNAMIC_BITSET_BITHACKS_H

template<int i>
int bitCount(u64 val){

}
//
//template<typename T>
//int countBits_LUT(T v){
//
//    return LUT_HW_16[v & 0xffff] + LUT_HW_16[(v >> 16) & 0xffff] + LUT_HW_16[(v >> 32) & 0xffff]+LUT_HW_16[(v >> 48) & 0xffff];
//}

template<typename T>
int countBits(T v) {
    const int CHAR_BIT = 8;
    const u64 k1 = 0x5555555555555555ull; /*  -1/3   */
    const u64 k2 = 0x3333333333333333ull; /*  -1/5   */
    const u64 k4 = 0x0f0f0f0f0f0f0f0full; /*  -1/17  */
    const u64 kf = 0x0101010101010101ull; /*  -1/255 */

    v = v - ((v >> 1) & k1);                           // temp
    v = (v & k2) + ((v >> 2) & k2);      // temp
    v = (v + (v >> 4)) & k4;                      // temp
    return (v * kf) >> (sizeof(T) - 1) * CHAR_BIT; // count
}

int NumberOfSetBits(int i)
{
    // Java: use >>> instead of >>
    // C or C++: use uint32_t
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

template<typename T>
int popCount (T x) {
    const u64 k1 = 0x5555555555555555ull; /*  -1/3   */
    const u64 k2 = 0x3333333333333333ull; /*  -1/5   */
    const u64 k4 = 0x0f0f0f0f0f0f0f0full; /*  -1/17  */
    const u64 kf = 0x0101010101010101ull; /*  -1/255 */

    x =  x       - ((x >> 1)  & k1); /* put count of each 2 bits into those 2 bits */
    x = (x & k2) + ((x >> 2)  & k2); /* put count of each 4 bits into those 4 bits */
    x = (x       +  (x >> 4)) & k4 ; /* put count of each 8 bits into those 8 bits */
    x = (x * kf) >> 56; /* returns 8 most significant bits of x + (x<<8) + (x<<16) + (x<<24) + ...  */
    return (int) x;
}
#endif //DYNAMIC_BITSET_BITHACKS_H
