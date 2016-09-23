//
// Created by Dusan Klinec on 16.06.16.
//

#include <iostream>
#include "dynamic_bitset.h"
#include "bit_array.h"
#include <random>
#include <fstream>
#include "CommonFnc.h"
#include "TermGenerator.h"

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

// Utilities for min-heap in array
void push_min_heap(vector<pairZscoreTerm>& heap, pairZscoreTerm val) {
    heap.push_back(val);
    push_heap(heap.begin(), heap.end(), pairDiffTermCompare());
}

pairZscoreTerm pop_min_heap(vector<pairZscoreTerm>& heap) {
    pairZscoreTerm val = heap.front();

    //This operation will move the smallest element to the end of the vector
    pop_heap(heap.begin(), heap.end(), pairDiffTermCompare());

    //Remove the last element from vector, which is the smallest element
    heap.pop_back();
    return val;
}


bool all_combinations(vector<int>& com, int n) {
    static int counter = 0;
    int tmp, ind = 0;
    counter++;
    if (counter == (1 << n)) {
        counter = 0;
        return false;
    }

    tmp = counter;
    com.clear();

    while(tmp)
    {
        if (tmp & 1)com.push_back(ind);
        ind++;
        tmp >>= 1;
    }

    return true;
}

void addSetVars(set<int>& S, vector<int>& term) {
    for (size_t i = 0; i < term.size(); i++)
    {
        S.insert(term[i]);
    }
}

bool subsetVars(set<int>& S, vector<int>& term) {
    for (size_t i = 0; i < term.size(); i++)
    {
        if (S.find(term[i]) == S.end()) return false;
    }
    return true;
}

double expProbofXorTerms(std::vector<pairZscoreTerm>& termsForXoring, int numVars){
    set<int> S;
    vector<int> com;
    double prob = 0;
    int ind;

    int numTerms = termsForXoring.size();
    while (all_combinations(com, numTerms)) {
        //print(com);
        S.clear();

        for(int i = 0; i < com.size(); i++)
        {
            ind = com[i];
            addSetVars(S, termsForXoring[ind].second);
        }
        if(com.size() & 1 ) prob += 1.0 / (1 << (S.size()-com.size()+1));
        else prob -= 1.0 / (1 << (S.size()-com.size()+1));
    }

    return prob;
}

double expProbofANDTerms(std::vector<pairZscoreTerm>& termsForAnding, int numVars){
    set<int> S;
    for(int i = 0; i < termsForAnding.size(); i++)
    {
        addSetVars(S, termsForAnding[i].second);
    }

    return 1.0 / (1 << S.size());
}


double XORkbestTerms(vector<bitarray<u64> >& bestTermsEvaluations, vector<pairZscoreTerm>& bestTerms, int k, int numVars, int numTVs, vector<int>& best_combination){
    vector<int> combination;
    int numTerms = bestTerms.size(), termInd, freqOnes;
    bitarray<u64> XORedBitarrays(numTVs);
    vector<pairZscoreTerm> termsForXoring;
    double expectedProb, observedProb,zscore,biggestZscore = 0;

    init_comb(combination, k);
    do {
        XORedBitarrays.reset();
        termsForXoring.clear();

        for (int i = 0; i < combination.size(); ++i) {
            termInd = combination[i];
            termsForXoring.push_back(bestTerms[termInd]);
            XOR(XORedBitarrays,XORedBitarrays,bestTermsEvaluations[termInd]);
        }
        freqOnes = HW(XORedBitarrays);
        observedProb = (double)freqOnes/numTVs;
        expectedProb = expProbofXorTerms(termsForXoring, numVars);
        zscore = CommonFnc::zscore(observedProb,expectedProb,numTVs);
        if(biggestZscore < zscore){
            biggestZscore = zscore;
            best_combination = combination;
        }
    }while(next_combination(combination,numTerms,false) );

    return biggestZscore;
}

double ANDkbestTerms(vector<bitarray<u64> >& bestTermsEvaluations, vector<pairZscoreTerm>& bestTerms, int k, int numVars, int numTVs){
    vector<int> combination;
    int numTerms = bestTerms.size(), termInd, freqOnes;
    bitarray<u64> ANDedBitarrays(numTVs);
    vector<pairZscoreTerm> termsForANDing;
    double expectedProb, observedProb,zscore,biggestZscore = 0;

    init_comb(combination, k);
    do {
        ANDedBitarrays.reset();
        termsForANDing.clear();

        for (int i = 0; i < combination.size(); ++i) {
            termInd = combination[i];
            termsForANDing.push_back(bestTerms[termInd]);
            XOR(ANDedBitarrays,ANDedBitarrays,bestTermsEvaluations[termInd]);
        }
        freqOnes = HW(ANDedBitarrays);
        observedProb = (double)freqOnes/numTVs;
        expectedProb = expProbofANDTerms(termsForANDing, numVars);
        zscore = CommonFnc::zscore(observedProb,expectedProb,numTVs);

        if(biggestZscore < zscore) biggestZscore = zscore;
    }while(next_combination(combination,numTerms,false) );

    return biggestZscore;
}
