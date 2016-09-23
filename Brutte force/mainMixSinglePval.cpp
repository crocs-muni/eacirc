//
// Created by syso on 9/22/2016.
//

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

    const int numTVs = 10000;
    const int tvsize = 16, numVars = 8 * tvsize;
    const int numEpochs = 40;
    const int numBytes = numTVs * tvsize;
    const int deg = 2;
    const int maxTerms = 30;
    u8 *TVs = new u8[numBytes];


    int Nr = 12;
    //string path = "C:/Users/syso/.CLion2016.1/system/cmake/generated/Brutte force-2fe67558/2fe67558/Release/";

    string filePath =  "MD6_CRT_" + std::to_string(Nr) + ".bin";
    cout << filePath << endl;
    ifstream in(filePath.c_str(), ios::binary);



    int resultSize = TarraySize<u64>(numTVs);
    u64 *block = new u64[resultSize * 128];
    vector < bitarray < u64 > * > resultArrays;

    // s = base vector. Evaluation of x_i on the input data.
    // The term of 3 variables is then evaluated by: s[i]&s[j]&s[k].
    SimpleTerm <u64, u64> s[numVars];
    for (int j = 0; j < numVars; ++j) {
        s[j].alloc(numVars);
        s[j].set(j);
        s[j].allocResults(numTVs);
        resultArrays.push_back(&s[j].getResults());
    }

    double pval;

    // Min-heap of best maxTerms terms
    vector<pairZscoreTerm> bestTerms(maxTerms);

    // Evaluated best terms on the data. Used for combining terms into polynomials (faster eval)
    vector<bitarray<u64> > bestTermsEvaluations(maxTerms);
    vector<hwres> bestTermsHw(maxTerms);
    for (int j = 0; j < maxTerms; ++j) {
        bestTermsEvaluations[j].alloc(numTVs);
    }

    in.read((char *) TVs, numBytes);


    for (int j = 0; j < numVars; ++j) {
        s[j].evaluateTVs(tvsize, TVs);
    }

    // Setting diff values to -1 is enough as they will got replaced by
    // non-negative ones in the next process in min-heap insertion.
    for(int tmpIdx=0; tmpIdx < maxTerms; ++tmpIdx){
        bestTerms[0].first = -1;
        bestTermsEvaluations[0].reset();
    }

    // Compute top K best distinguishers
    // TODO: in order to test the hypothesis about TOP k best distinguishers
    // replace the logic inside this method so it picks random k distinguishers.
    // If our hypothesis works, the results with "best" k distinguishers" should be better
    // than with "random" k distinguishers.
    computeTopKInPlace<deg>(resultArrays, bestTerms, maxTerms, numTVs);

    for(int i = 0; i < bestTerms.size(); i++)
        cout << bestTerms[i].first << endl;
    return 0;
}