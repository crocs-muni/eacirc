//
// Created by Dusan Klinec on 16.06.16.
//

#ifndef BRUTTE_FORCE_TERMGENERATOR_H
#define BRUTTE_FORCE_TERMGENERATOR_H
#include <vector>
#include <stdexcept>

void init_comb(std::vector<int> &com, int k);

bool next_combination(std::vector<int> &com, int max, bool disjoint = false);

float Chival(int diff, int deg, int numTVs);

void genRandData(u8 *TVs, int numBytes);

template<typename T = uint64_t>
inline T serializeTerm(std::vector<int> &indices, int deg){
    if (deg > sizeof(T)){
        throw std::out_of_range("term too big to serialize");
    }

    T res = 0;
    for(int i=0;i<deg;++i){
        if (indices[i] > 0xff){
            throw std::out_of_range("term var too big to serialize");
        }

        res |= ((T)(indices[i]&0xff)) << (8*i);
    }
    return res;
}

template<typename T = uint64_t>
inline void deserializeTerm(std::vector<int> &indices, int deg, T term){
    if (deg > sizeof(T)){
        throw std::out_of_range("term too big to serialize");
    }
    for(int i=0;i<deg;++i){
        indices[i] = (term >> (8*i))&0xff;
    }
}

template<class T>
void printVec(std::vector<T> v) {
    for (int i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;

}

template<int deg>
int compute(std::vector<bitarray<u64> * > a,
            std::vector<int> &bestTerm,
            int numTVs)
{
    int diff, biggestDiff = 0;
    const int refCount = numTVs >> deg;
    std::vector<int> indices;

    init_comb(indices, deg);
    do {
        diff = abs(HW_AND<deg>(a, indices) - refCount);
        if (biggestDiff < diff) {
            bestTerm = indices;
            biggestDiff = diff;
        }

    } while (next_combination(indices, 128));

    return biggestDiff;
}


#endif //BRUTTE_FORCE_TERMGENERATOR_H
