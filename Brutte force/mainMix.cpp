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
    const int maxTerms = 30;
    u8 *TVs = new u8[numBytes];

    ifstream in(argv[1], ios::binary);

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

    int diff;
    double chisqrValue, pval;

    vector<int> bestTermIndices;
    vector<double> Pvals;

    // Min-heap of best maxTerms terms
    vector<pairDiffTerm> bestTerms(maxTerms);
    // Evaluated best maxTerms terms, used for further processing (base).
    vector<Term <u64>> evaluatedBestTerms(maxTerms);

    in.read((char *) TVs, numBytes);
    //genRandData(TVs, numBytes);

    int counter = numBytes;
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        for (int j = 0; j < numVars; ++j) {
            s[j].evaluateTVs(tvsize, TVs);
        }

        // Reset best terms from the previous round.
        // Setting diff values to -1 is enough as they will got replaced by
        // non-negative ones in the next process in min-heap insertion.
        for(int tmpIdx=0; tmpIdx < maxTerms; ++tmpIdx){
            bestTerms[0].first = -1;
        }

        // Compute top K best distinguishers
        // TODO: in order to test the hypothesis about TOP k best distinguishers
        // replace the logic inside this method so it picks random k distinguishers.
        // If our hypothesis works, the results with "best" k distinguishers" should be better
        // than with "random" k distinguishers.
        computeTopKInPlace<deg>(resultArrays, bestTerms, maxTerms, numTVs);

        // Now you can get top K max diff terms by order - by calling k * pop() / pop_min_heap().
        // Or use the following trick to extract the underlying data structure from the
        // priority_queue.

        // Marek, if you want, we can read a new data set here and evaluate...
        in.read((char *) TVs, numBytes);

        // -------------------------------------------------------------------------------------------------------------
        // Evaluate top best terms k on new data.
        // The evaluation is helpful for computing fitness of the term itself AND for the further evaluation
        // of pairs, triplets, ... n-tuples of terms (polynomials).
        for(unsigned termIdx = 0; termIdx < maxTerms; ++termIdx){
            evaluatedBestTerms[0] = Term<u64>(128, bestTerms[termIdx].second);
            int curResults = evaluatedBestTerms[0].evaluateTVs(tvsize, numTVs, TVs);

            // Log if you want ;)
            diff = abs(curResults - (numTVs >> deg));

            // The old evaluation routine for terms only
            chisqrValue = Chival(diff, deg, numTVs);
            pval = CommonFnc::chisqr(1, chisqrValue);
            cout << " term-only-difference:= " << diff << "  p-value:= " << pval << endl;
        }

        // Compute all pairs on terms
        // Here we use simple nested for. In general the same logic
        // as next_combination() uses can be used, if needed, just one level above.
        for(unsigned termIdx1 = 0; termIdx1 < maxTerms-1; ++termIdx1){
            for(unsigned termIdx2 = termIdx1 + 1; termIdx2 < maxTerms; ++termIdx2){
                // XOR

            }
        }

        //Pvals.push_back(pval);
    }

    Logger logger{"eacirc.log"};
    printVec(Pvals);
    Finisher::ks_test_finish(Pvals, 5);


    return 0;
}