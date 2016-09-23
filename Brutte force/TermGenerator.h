//
// Created by Dusan Klinec on 16.06.16.
//

#ifndef BRUTTE_FORCE_TERMGENERATOR_H
#define BRUTTE_FORCE_TERMGENERATOR_H
#include <vector>
#include <stdexcept>
#include <queue>
#include "base.h"

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
void printVec(std::vector<T> v) {
    for (int i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;

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

// diff from the expected value of 'ones' + term
typedef std::pair<int, term> pairDiffTerm;

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
    bool operator()(const pairDiffTerm& l, const pairDiffTerm& r)
    {
        return l.first > r.first;
    }
};

// Priority queue
typedef std::priority_queue<pairDiffTerm, std::vector<pairDiffTerm>, pairDiffTermCompare> priorityQueueOnTerms;

template<int deg>
int computeTopK(std::vector<bitarray<u64> * > a,
               priorityQueueOnTerms &queue,
                int maxTerms,
                int numTVs)
{
    int diff, biggestDiff = 0;
    const int refCount = numTVs >> deg;
    std::vector<int> indices;

    // Keeping top K max elements in the priority queue.
    // Priority queue is min-heap. If new element is lower than minimum
    // then ignore it. Otherwise add it to the heap and delete the previous minimum.
    init_comb(indices, deg);
    do {
        diff = abs(HW_AND<deg>(a, indices) - refCount);

        // If queue is not full OR the value is higher than queue-minimal, add it.
        const pairDiffTerm & c_top = queue.top();
        if (queue.size() < maxTerms || diff > c_top.first){
            pairDiffTerm c_pair(diff, indices);
            queue.push(c_pair);

            if (queue.size() > maxTerms){
                queue.pop();
            }
        }
    } while (next_combination(indices, 128));

    return biggestDiff;
}

// In place variant without reallocations.
void push_min_heap(std::vector<pairDiffTerm>& heap, pairDiffTerm val);
pairDiffTerm pop_min_heap(std::vector<pairDiffTerm>& heap);

template<int deg>
int computeTopKInPlace(std::vector<bitarray<u64> * > a,
                std::vector<pairDiffTerm> &queue,
                int maxTerms,
                int numTVs,
                int tvsize = 128)
{
    int diff, biggestDiff = 0;
    const int refCount = numTVs >> deg;
    std::vector<int> indices;

    // Make sure the queue is of the given size.
    queue.resize((unsigned)maxTerms, pairDiffTerm(-1, term()));

    // Keeping top K max elements in the priority queue.
    // Priority queue is min-heap. If new element is lower than minimum
    // then ignore it. Otherwise add it to the heap and delete the previous minimum.
    init_comb(indices, deg);
    //init queue
    //put first maxTerms of terms to queue
    for (int i = 0; i < maxTerms; ++i) {
        diff = abs(HW_AND<deg>(a, indices) - refCount);
        next_combination(indices, tvsize);
        pairDiffTerm c_pair(diff, indices);
        push_min_heap(queue, c_pair);
    }
    do {
        diff = abs(HW_AND<deg>(a, indices) - refCount);

        // If queue is not full OR the value is higher than queue-minimal, add it.
        const pairDiffTerm & c_top = queue.back();
        if (queue.size() < maxTerms || diff > c_top.first){
            pairDiffTerm c_pair(diff, indices);
            push_min_heap(queue, c_pair);

            if (queue.size() > maxTerms){
                pop_min_heap(queue);
            }
        }
    } while (next_combination(indices, tvsize));

    return biggestDiff;
}


#endif //BRUTTE_FORCE_TERMGENERATOR_H
