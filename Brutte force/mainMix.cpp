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

    vector<double> Pvals;

    // Min-heap of best maxTerms terms
    vector<pairDiffTerm> bestTerms(maxTerms);

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

    //genRandData(TVs, numBytes);

    int counter = numBytes;
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        // Reset best terms from the previous round.
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

        // Now you can get top K max diff terms by order - by calling k * pop() / pop_min_heap().
        // Or use the following trick to extract the underlying data structure from the
        // priority_queue.

        // If you want, we can read a new data set here and evaluate...
        in.read((char *) TVs, numBytes);

        // Data changed -> re-evaluate base terms / regenerate basis
        // This is needed for further fast evaluation of combined terms.
        for (int j = 0; j < numVars; ++j) {
            s[j].evaluateTVs(tvsize, TVs);
        }

        // -------------------------------------------------------------------------------------------------------------
        // Evaluate top best terms k on new data.
        // The evaluation is helpful for computing fitness of the term itself.
        for(unsigned termIdx = 0; termIdx < maxTerms; ++termIdx){
            // We have basis regenerated now, we can evaluate terms on new data faster with using the basis.
            // The evaluation result is stored to bestTermsEvaluations for further combinations.
            const int result_hw = HW_AND(bestTermsEvaluations[termIdx], resultArrays, bestTerms[termIdx].second);
            bestTermsHw[termIdx] = result_hw;

            diff = abs(result_hw - (numTVs >> deg));

            // The old evaluation routine for terms only
            chisqrValue = Chival(diff, deg, numTVs);
            pval = CommonFnc::chisqr(1, chisqrValue);

            printf(" term-only-difference: %d, old-data-diff: %d, p-value:=%f\n", diff, bestTerms[termIdx].first, pval);
            //cout << " term-only-difference:= " << diff << "  p-value:= " << pval << endl;
        }

        // Compute all pairs on terms
        // Here we use simple nested for. In general the same logic
        // as next_combination() uses can be used, if needed, just one level above.
        for(unsigned termIdx1 = 0; termIdx1 < maxTerms-1; ++termIdx1){
            for(unsigned termIdx2 = termIdx1 + 1; termIdx2 < maxTerms; ++termIdx2){
                const hwres term1_hw = bestTermsHw[termIdx1];
                const hwres term2_hw = bestTermsHw[termIdx2];

                // termIdx1 XOR termIdx2
                const hwres xor_hw = HW_XOR(bestTermsEvaluations[termIdx1], bestTermsEvaluations[termIdx2]);

                // termIdx1 AND termIdx2
                const hwres and_hw = HW_AND(bestTermsEvaluations[termIdx1], bestTermsEvaluations[termIdx2]);

                // Sorry for printf, its just better then cout in this case...
                // WARNING - hardcoded term deg = 3, you need to change that.
                // But just to visualize how various the terms are...
                printf(" - [%03u, %03u], hwTerm1: %d, hwTerm2: %d, hwXor: %d, hwAnd: %d "
                               "t1: %d,%d,%d t2: %d,%d,%d\n",
                    termIdx1, termIdx2, (int)term1_hw, (int)term2_hw, (int)xor_hw, (int)and_hw,
                    bestTerms[termIdx1].second[0],
                    bestTerms[termIdx1].second[1],
                    bestTerms[termIdx1].second[2],
                    bestTerms[termIdx2].second[0],
                    bestTerms[termIdx2].second[1],
                    bestTerms[termIdx2].second[2]
                );
            }
        }

        //Pvals.push_back(pval);
    }

    Logger logger{"eacirc.log"};
    printVec(Pvals);
    Finisher::ks_test_finish(Pvals, 5);


    return 0;
}