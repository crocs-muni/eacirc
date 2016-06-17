//
// Created by Dusan Klinec on 16.06.16.
//

#include "mainBi.h"

#include <iostream>
#include "dynamic_bitset.h"
#include "bit_array.h"
#include "Term.h"
#include "bithacks.h"
#include <ctime>
#include <random>
#include <fstream>
#include <unordered_map>
#include "CommonFnc.h"
#include "finisher.h"
#include "logger.h"
#include "TermGenerator.h"
#include <algorithm>
#include <string>
#include <unordered_map>

using namespace std;
#define TERM_WIDTH_BYTES 16
#define TERM_DEG 3
//#define DUMP_ZSCORE 1

// Do not edit.
#define TERM_WIDTH (TERM_WIDTH_BYTES*8)
// Combination(TERM_WIDTH, TERM_DEG) for C(128,3)
#define TERM_NUMBER 341376
typedef std::vector<int> termRep;

/**
 * Test if x_i x_j x_k and x_i x_j x_l where i!=j!=k!=l can skew statistics
 * due to common i, j sub-terms.
 *
 * Does by simulating the exact calculation.
 */
int testDeps(){
    return -1; // TODO:impl.
}

template<typename T>
void histogram(vector<T> data, unsigned long bins, bool center = false){
    const unsigned long size = data.size();
    const double mean = accumulate(data.begin(), data.end(), 0) / (double)size;
    T min = *(min_element(data.begin(), data.end()));
    T max = *(max_element(data.begin(), data.end()));
    double binSize = (max-min)/(double)bins;

    // Center around mean, so mean is in the middle of the bin.
    if (center){
        const double meanBinLow = mean-binSize/2;
        const double binsMeanLeft = ceil((meanBinLow - min) / binSize);
        const double newMin = meanBinLow - binsMeanLeft * binSize;
        assert(newMin <= min);
        min = newMin;
    }

    printf("(hist binSize: %0.6f, min: %0.6f, mean: %0.6f, max: %0.6f)\n", binSize, min, mean, max);

    vector<unsigned long> binVector(bins+2);
    fill(binVector.begin(), binVector.end(), 0);

    for(int i = 0; i<size; i++){
        binVector[ (data[i] - min)/binSize ] += 1;
    }

    unsigned long binMax = *(max_element(binVector.begin(), binVector.end()));
    for(int i = 0; i < bins; i++){
        printf("%04d[c:%+0.6f]: ", i, min + binSize*i + binSize/2.0);
        int chars = (int)ceil(100 * (binVector[i] / (double)binMax));
        for(int j=0; j<chars; j++){
            cout << "+";
        }
        cout << endl;
    }
}

/**
 * Compute
 */
int testBi(ifstream &in){
    const int numTVs = 1024*512; // keep this number divisible by 128 pls!
    const int numEpochs = 1;
    const int numBytes = numTVs * TERM_WIDTH_BYTES;
    const bool disjointTerms = false;

    u8 *TVs = new u8[numBytes];
    vector < bitarray < u64 > * > resultArrays;
    SimpleTerm <u64, u64> s[TERM_WIDTH];

    // Allocation, initialization.
    for (int j = 0; j < TERM_WIDTH; ++j) {
        s[j].alloc(TERM_WIDTH);
        s[j].set(j);
        s[j].allocResults(numTVs);
        resultArrays.push_back(&s[j].getResults());
    }

    // Remembers all results for all polynomials.
    // unordered_map was here before, but we don't need it for now as
    // order on polynomials is well defined for given order - by the generator.
    u64 termCnt = 0;
    vector<u64> resultStats(TERM_NUMBER);
    for(u64 idx = 0; idx < TERM_NUMBER; idx++){
        resultStats[idx] = 0;
    }

    // Epoch is some kind of redefined here.
    // Epoch = next processing step of the input data of size numBytes.
    // Statistical processing is done over all epochs. It makes some kind of trade-off between CPU/RAM.
    // If desired, wrap this with another level of loop.
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        printf("## EPOCH: %02d\n", epoch);

        // Read test vectors.
        in.read((char *) TVs, numBytes);

        // Single-var term s_j (hw(s_j)=1) is evaluated numTVs times on 128 input bits
        for (int j = 0; j < TERM_WIDTH; ++j) {
            s[j].evaluateTVs(TERM_WIDTH_BYTES, TVs);
        }
        printf("  elementary results computed\n");

        // Generate all polynomials from precomputed values.
        termRep indices;
        u64 termIdx = 0;
        init_comb(indices, TERM_DEG);
        do {
            // Number of times the polynomial <indices> returned 1 on 128bit test vector.
            int hw = HW_AND<TERM_DEG>(resultArrays, indices);
            resultStats[termIdx++] += (u64)hw;

        } while (next_combination(indices, TERM_WIDTH, disjointTerms));
        if (termCnt == 0) termCnt = termIdx;
    }

    // Result processing.
    double expectedOccurrences = (numTVs * numEpochs) / (double)(1 << TERM_DEG);
    double expectedProb = 1.0 / (1 << TERM_DEG);
    printf("Expected occ: %.6f, expected prob: %.6f\n", expectedOccurrences, expectedProb);

    // Output file with z-scores.
#ifdef DUMP_ZSCORE
    ofstream scoreFile("./zscores.csv", ios::trunc);
    scoreFile << "polyIdx;zscore" << endl;
#endif
    vector<double> zscores(termCnt);
    u64 polyTotalCtr = 0;
    u64 totalObserved = 0;
    u64 rejected95 = 0;
    u64 rejected99 = 0;
    double zscoreTotal = 0;
    termRep indices;
    init_comb(indices, TERM_DEG);
    do {
        u64 observed = resultStats[polyTotalCtr];
        totalObserved += observed;

        double observedProb = (double)observed / numTVs / numEpochs;
        double zscore = CommonFnc::zscore(observedProb, expectedProb, numTVs * numEpochs);
        double zscoreAbs = abs(zscore);
        if (zscoreAbs > 1.96){
            rejected95+=1;
        }
        if (zscoreAbs > 2.576){
            rejected99+=1;
        }

        zscores[polyTotalCtr] = zscore;
        zscoreTotal += zscoreAbs;
        if (polyTotalCtr < 128){
            printf("Observed[%08x]: %08llu, probability: %.6f, z-score: %0.6f\n",
                   (unsigned)polyTotalCtr, observed, observedProb, zscore);
        }

#ifdef DUMP_ZSCORE
        scoreFile << polyTotalCtr << ";" << zscore << endl;
#endif

        polyTotalCtr+=1;
    } while (next_combination(indices, TERM_WIDTH, disjointTerms));

#ifdef DUMP_ZSCORE
    // Finish zscore file.
    scoreFile.close();
#endif

    double avgOcc = (double)totalObserved / polyTotalCtr;
    double avgProb = avgOcc / (numTVs * numEpochs);

    printf("z-score histogram: \n");
    histogram(zscores, 51, true);

    printf("Done, totalTerms: %04llu, acc: %08llu, average occurrence: %0.6f, average prob: %0.6f\n",
           polyTotalCtr, totalObserved, avgOcc, avgProb);

    printf("      ztotal: %0.6f, avg-zscore: %0.6f\n", zscoreTotal, zscoreTotal/polyTotalCtr);

    printf("# of rejected 95%%: %04llu that is %0.6f%%\n", rejected95, 100.0*rejected95/polyTotalCtr);
    printf("# of rejected 99%%: %04llu that is %0.6f%%\n", rejected99, 100.0*rejected99/polyTotalCtr);

    // Test Binomial distribution hypothesis.

    return 0;
}



int main(int argc, char *argv[]) {
    if (argc < 2){
        printf("No input file given");
        return -1;
    }

    printf("Opening file: %s\n", argv[1]);
    ifstream in(argv[1], ios::binary);
    testBi(in);

    return 0;
}