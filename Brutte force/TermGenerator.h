//
// Created by Dusan Klinec on 16.06.16.
//

#ifndef BRUTTE_FORCE_TERMGENERATOR_H
#define BRUTTE_FORCE_TERMGENERATOR_H
#include <vector>
#include <stdexcept>
#include <queue>
#include <vector>
#include <set>
#include "base.h"
#include "CommonFnc.h"
#include <fstream>

void init_comb(std::vector<int> &com, int k);

bool next_combination(std::vector<int> &com, int max, bool disjoint = false);

float Chival(int diff, int deg, int numTVs);

void genRandData(u8 *TVs, int numBytes);

template<typename T = uint64_t>
inline T serializeTerm(std::vector<int> &indices, int deg){
    if (deg > sizeof(T)){
        throw std::out_of_range("term too big to serialize");
    }

    T res = 0;
    for(int i=0;i<deg;++i){
        if (indices[i] > 0xff){
            throw std::out_of_range("term var too big to serialize");
        }

        res |= ((T)(indices[i]&0xff)) << (8*i);
    }
    return res;
}

template<typename T = uint64_t>
inline void deserializeTerm(std::vector<int> &indices, int deg, T term){
    if (deg > sizeof(T)){
        throw std::out_of_range("term too big to serialize");
    }
    for(int i=0;i<deg;++i){
        indices[i] = (term >> (8*i))&0xff;
    }
}

template<class T>
void printVec(std::vector<T> v, bool newline = true, std::ostream& out = std::cout) {
    for (int i = 0; i < v.size(); ++i) {
        out << v[i] << " ";
    }
    if (newline)out << std::endl;
}

template<int deg>
int compute(std::vector<bitarray<u64> * > a,
            std::vector<int> &bestTerm,
            int numTVs)
{
    int diff, biggestDiff = 0;
    const int refCount = numTVs >> deg;
    std::vector<int> indices;

    init_comb(indices, deg);
    do {
        diff = abs(HW_AND<deg>(a, indices) - refCount);
        if (biggestDiff < diff) {
            bestTerm = indices;
            biggestDiff = diff;
        }

    } while (next_combination(indices, 128));

    return biggestDiff;
}

// zscore from the expected value of 'ones' + term
typedef std::pair<double, term> pairZscoreTerm;

// Utility which helps us to extract underlying container from the std::priority_queue.
template <class T, class S, class C>
S& Container(std::priority_queue<T, S, C>& q) {
    struct HackedQueue : private std::priority_queue<T, S, C> {
        static S& Container(std::priority_queue<T, S, C>& q) {
            return q.*&HackedQueue::c;
        }
    };
    return HackedQueue::Container(q);
}

// Comparator on pairDiffTerm so we have min-heap.
struct pairDiffTermCompare
{
    bool operator()(const pairZscoreTerm& l, const pairZscoreTerm& r)
    {
        return l.first > r.first;
    }
};

// Priority queue
typedef std::priority_queue<pairZscoreTerm, std::vector<pairZscoreTerm>, pairDiffTermCompare> priorityQueueOnTerms;


// In place variant without reallocations.
void push_min_heap(std::vector<pairZscoreTerm>& heap, pairZscoreTerm val);
pairZscoreTerm pop_min_heap(std::vector<pairZscoreTerm>& heap);

template<int deg>
double computeTopKInPlace(std::vector<bitarray<u64> * > a,
                std::vector<pairZscoreTerm> &queue,
                int maxTerms,
                int numTVs,
                int numVars = 128)
{
    double zscore, biggestZscore = 0;
    const int refCount = numTVs >> deg;
    std::vector<int> indices;
    int freqOnes;

    // Make sure the queue is of the given size.
    queue.resize((unsigned)maxTerms, pairZscoreTerm(-1, term()));

    // Keeping top K max elements in the priority queue.
    // Priority queue is min-heap. If new element is lower than minimum
    // then ignore it. Otherwise add it to the heap and delete the previous minimum.
    init_comb(indices, deg);
    do {
        freqOnes = HW_AND<deg>(a, indices);
        zscore = CommonFnc::zscore((double)freqOnes/numTVs,(double)refCount/numTVs,numTVs);
        // If queue is not full OR the value is higher than queue-minimal, add it.
        const pairZscoreTerm & c_top = queue.back();
        if (queue.size() < maxTerms || zscore > c_top.first){
            pairZscoreTerm c_pair(zscore, indices);
            push_min_heap(queue, c_pair);

            if (queue.size() > maxTerms){
                pop_min_heap(queue);
            }
        }
    } while (next_combination(indices, numVars));

    return biggestZscore;
}

bool all_combinations(std::vector<int>& com, int n);

double expProbofXORTerms(std::vector<pairZscoreTerm>& termsForXoring,  int tvsize = 128);
double expProbofANDTerms(std::vector<pairZscoreTerm>& termsForAnding,  int tvsize = 128);

double XORkbestTerms(std::vector<bitarray<u64> >& bestTermsEvaluations, std::vector<pairZscoreTerm>& bestTerms, int k, int numVars, int numTVs, std::vector<int>& best_combination);
double ANDkbestTerms(std::vector<bitarray<u64> >& bestTermsEvaluations, std::vector<pairZscoreTerm>& bestTerms, int k, int numVars, int numTVs);
#endif //BRUTTE_FORCE_TERMGENERATOR_H
