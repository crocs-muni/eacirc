//
// Created by Dusan Klinec on 16.06.16.
//

#ifndef BRUTTE_FORCE_TERMGENERATOR_H
#define BRUTTE_FORCE_TERMGENERATOR_H
#include <vector>

void init_comb(std::vector<int> &com, int k);

bool next_combination(std::vector<int> &com, int max);

float Chival(int diff, int deg, int numTVs);

void genRandData(u8 *TVs, int numBytes);

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
