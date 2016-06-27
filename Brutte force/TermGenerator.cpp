//
// Created by Dusan Klinec on 16.06.16.
//

#include <iostream>
#include "dynamic_bitset.h"
#include "bit_array.h"
#include <random>
#include <fstream>
#include "CommonFnc.h"

using namespace std;

void init_comb(vector<int> &com, int k) {
    com.clear();
    for (int i = 0; i < k; ++i) {
        com.push_back(i);
    }
}

bool next_combination(vector<int> &com, int max, bool disjoint) {
    const int size = (int)com.size();
    int idx = size - 1;

    // disjoint: 0, 1, 2 -> 3, 4, 5
    if (disjoint){
        if (com[idx] + size >= max) return false;
        for(idx=0; idx < size; ++idx){
            com[idx] += size;
        }

        return true;
    }

    if (com[idx] == max - 1) {
        do {
            idx -= 1;
        } while (idx >= 0 && com[idx] + 1 == com[idx + 1]);

        if (idx < 0) {
            return false;
        }

        for (int j = idx + 1; j < size; ++j) {
            com[j] = com[idx] + j - idx + 1;
        }
    }
    com[idx]++;
    return true;
}

float Chival(int diff, int deg, int numTVs) {
    int refCount = (numTVs >> deg);
    return ((float) diff * diff / refCount) + ((float) diff * diff / (numTVs - refCount));
}

void genRandData(u8 *TVs, int numBytes) {
    for (size_t i = 0; i < numBytes; i++) {
        TVs[i] = rand();
    }
}

