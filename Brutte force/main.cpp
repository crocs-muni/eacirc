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
#include "TermGenerator.h"

using namespace std;

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