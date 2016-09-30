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
#include <iomanip>
using namespace std;

int main(int argc, char *argv[]) {
    const int deg = 2;
    int numTVs = 10000000;
    int tvsize = 16, numVars = 8 * tvsize;
    int kbound = 2;

    ifstream in(argv[1], ios::binary);
    int maxTerms = atoi(argv[2]);
    //tvsize = atoi(argv[3])/8;
    numTVs = atoi(argv[4]);

    if(maxTerms == 10) kbound = 10;
    else kbound = 2;

    int numBytes = numTVs * tvsize;
    u8 *TVs = new u8[numBytes];

    long long input_size = CommonFnc::getFileSize(argv[1]);
    if (numBytes > input_size) {
        cerr << "Input file " << argv[1] << " has " << input_size << " B but required is " << numBytes << endl;
        return -1;
    }

    ofstream ZscoreFile("best Zscore.txt",std::ofstream::app);
    ofstream besttermsFile("best Terms.txt",std::ofstream::app);


    int resultSize = TarraySize<u64>(numTVs);
    u64 *block = new u64[resultSize * 128];
    vector < bitarray < u64 > * > resultArrays;

    // s = base vector. Evaluation of x_i on the input data.
    // The term of 3 variables is then evaluated by: s[i]&s[j]&s[k].
    SimpleTerm <u64, u64>* s = new SimpleTerm <u64, u64>[numVars];
    for (int j = 0; j < numVars; ++j) {
        s[j].alloc(numVars);
        s[j].set(j);
        s[j].allocResults(numTVs);
        resultArrays.push_back(&s[j].getResults());
    }

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
        bestTerms[tmpIdx].first = -1;
        bestTermsEvaluations[tmpIdx].reset();
    }

    // Compute top K best distinguishers
    // TODO: in order to test the hypothesis about TOP k best distinguishers
    // replace the logic inside this method so it picks random k distinguishers.
    // If our hypothesis works, the results with "best" k distinguishers" should be better
    // than with "random" k distinguishers.
    computeTopKInPlace<deg>(resultArrays, bestTerms, maxTerms, numTVs);


    /*results << "best terms:" << endl;
    for(int i = 0; i < bestTerms.size(); i++) {
        results  <<  setw(3) << i << ". zscore=" << setprecision(3) <<  bestTerms[i].first << " [ ";
        printVec(bestTerms[i].second, false, results);
        results << "]\n";
    }*/

    ZscoreFile << argv[1] << " maxterms=" <<  maxTerms << " deg=" << deg << " NumTVs=10^"  << log10(numTVs) << " numVars=" << numVars;
    besttermsFile << argv[1] << " maxterms=" <<  maxTerms << " deg=" << deg << " NumTVs=10^"  << log10(numTVs) << " numVars=" << numVars;

    vector<int> best_combination;
    for (int k = 1; k < kbound+1; ++k) {
        double zscore = ANDkbestTerms(bestTermsEvaluations, bestTerms, k, numVars, numTVs, best_combination);

        ZscoreFile << " k= "  << k << " zscore= " <<  zscore << " ";
        besttermsFile << " k= "  << k << " zscore= " <<  zscore << "  best terms= ";

        for (int i = 0; i < best_combination.size(); ++i) {
            for (int j = 0; j < bestTerms[i].second.size(); ++j) {
                besttermsFile << bestTerms[i].second[j] << " ";
            }
            besttermsFile << "     ";
        }
        ZscoreFile << " " ;
        besttermsFile << " " ;
    }
    ZscoreFile << endl;
    besttermsFile <<endl;

    return 0;
}