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

bool next_combination(vector<int> &com, int max) {
    static int size = com.size();
    int idx = size - 1;

    if (com[idx] == max - 1) {
        while (com[--idx] + 1 == com[idx + 1]) {
            if (idx == 0) return false;
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

