//
// Created by syso on 16. 5. 2016.
//

#ifndef DYNAMIC_BITSET_TERM_H
#define DYNAMIC_BITSET_TERM_H

#include "dynamic_bitset.h"
#include "bit_array.h"
#include "bithacks.h"
#include <vector>

template <typename T = uint64_t >
class Term : public dynamic_bitset<T>{
public:
    Term(int numVars = 0, void* array = NULL):dynamic_bitset<T>(numVars,array){
    }

    Term(int numVars, std::vector<int> vars):dynamic_bitset<T>(numVars,NULL){
        for (int i = 0; i < vars.size(); ++i) {
            this->set(vars[i]);
        }
    }

    bool evaluate(void* array){
        T *pt = (T*)array;
        for (int i = 0; i < this->size(); ++i) {
            if( (pt[i] & (*this)[i]) != (*this)[i]) return false;
        }
        return true;
    }

    int evaluateTVs(int tvsize, int numTVs, void* data){
        u8* pt = (u8*)data;
        int counter = 0;
        for (int i = 0; i < numTVs; ++i) {
            if(this->evaluate(pt+i*tvsize))
                counter++;
        }
        return counter;
    }
private:
};

//simple terms of small degree
template <typename T = u64, typename Tres = u64 >
class SimpleTerm : public Term<T>{
public:
    SimpleTerm( int numVars = 0, void* array = NULL):Term<T>(numVars,array){
        _results = NULL;
    }
    void allocResults(int numTestVec){
        if(_results)delete _results;
        _results = new bitarray<Tres>(numTestVec);
    }

    void setResultArray(int numTestVec, void* result){
        if(_results)delete _results;
        _results = new bitarray<Tres>( numTestVec, result);
    }

    bitarray<Tres>& getResults(){
        return *_results;
    }

    void evaluateTVs(int tvsize, void* data){
        u8* pt = (u8*)data;
        for (int i = 0; i < _results->getNumVars(); ++i) {
			_results->set(i, this->evaluate(pt + i*tvsize));
        }
    }
    void printResults(){
        _results->print();
    }
private:
    bitarray<Tres>* _results;
};

template <typename T = u64, typename Tres = u64 >
int HW_AND(SimpleTerm<T,Tres>& t1, SimpleTerm<T,Tres>& t2){
    return HW_AND<Tres>(t1.getResults(), t2.getResults());
};

//template <typename T = u64, typename Tres = u64 >
//int HW_AND(SimpleTerm<T,Tres> t1, SimpleTerm<T,Tres> t2, SimpleTerm<T,Tres> t3){
//    return HW_AND<Tres>(t1.getResults(), t2.getResults(), t3.getResults());
//};

template <typename T = u64, typename Tres = u64 >
int HW_AND(SimpleTerm<T,Tres>& t1, SimpleTerm<T,Tres>& t2, SimpleTerm<T,Tres>& t3){
    int res = 0;
    for (int i = 0; i < t1.getResults().size(); ++i) {
        res +=  popCount(t1.getResults()[i] & t2.getResults()[i] & t3.getResults()[i]);
    }
    return  res;
}






typedef SimpleTerm<u64,u64> SimpleTerm64;


#endif //DYNAMIC_BITSET_TERM_H
