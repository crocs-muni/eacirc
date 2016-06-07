#include <iostream>
#include "dynamic_bitset.h"
#include "bit_array.h"
#include "Term.h"
#include "bithacks.h"
#include <ctime>
#include <random>
#include <fstream>
#include "CommonFnc.h"
#include "finisher.h"
#include "logger.h"

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

template<class T>
void printVec(vector <T> v) {
    for (int i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;

}

template<int deg>
int compute(vector<bitarray < u64> *

> a,
vector<int> &bestTerm,
int numTVs
){
int diff, biggestDiff = 0;
const int refCount = numTVs >> deg;
vector<int> indices;

init_comb(indices, deg
);
//printVec(indices);
do{
diff = abs(HW_AND<deg>(a, indices) - refCount);

if(biggestDiff<diff){
bestTerm = indices;
biggestDiff = diff;
}

}while (
next_combination(indices,
128));
//printVec(indices);

return
biggestDiff;
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

int main(int argc, char *argv[]) {

    const int numTVs = 100000;
    const int tvsize = 16, numVars = 8 * tvsize;
    const int numEpochs = 40;
    const int numBytes = numTVs * tvsize;
    const int deg = 3;
    u8 *TVs = new u8[numBytes];

    ifstream in(argv[1], ios::binary);

    int resultSize = TarraySize<u64>(numTVs);
    u64 *block = new u64[resultSize * 128];
    vector < bitarray < u64 > * > resultArrays;


    SimpleTerm <u64, u64> s[numVars];
    for (int j = 0; j < numVars; ++j) {
        s[j].alloc(numVars);
        s[j].set(j);
        s[j].allocResults(numTVs);
        resultArrays.push_back(&s[j].getResults());
    }

    int diff;
    double chisqrValue, pval;

    vector<int> bestTermIndices;
    vector<double> Pvals;

    in.read((char *) TVs, numBytes);
    //genRandData(TVs, numBytes);


    int counter = numBytes;

    for (int epoch = 0; epoch < numEpochs; ++epoch) {

        for (int j = 0; j < numVars; ++j) {
            s[j].evaluateTVs(tvsize, TVs);
        }
        //find best distinguisher
        diff = compute<deg>(resultArrays, bestTermIndices, numTVs);
        printVec(bestTermIndices);
        //test dist and compute p-val

        Term <u64> t(128, bestTermIndices);
        in.read((char *) TVs, numBytes);
        //genRandData(TVs, numBytes);
        diff = abs(t.evaluateTVs(tvsize, numTVs, TVs) - (numTVs >> deg));

        chisqrValue = Chival(diff, deg, numTVs);
        pval = CommonFnc::chisqr(1, chisqrValue);
        cout << " difference:= " << diff << "  p-value:= " << pval << endl;
        Pvals.push_back(pval);
    }

    Logger logger{"eacirc.log"};
    printVec(Pvals);
    Finisher::ks_test_finish(Pvals, 5);


    return 0;
}