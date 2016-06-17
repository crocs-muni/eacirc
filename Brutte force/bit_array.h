//
// Created by syso on 16. 5. 2016.
//

#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#include "dynamic_bitset.h"
#include "bithacks.h"
#include <vector>

template <typename T = uint64_t >
class bitarray : public dynamic_bitset<T>{
public:
    bitarray(int numVars = 0, void* array = NULL):dynamic_bitset<T>(numVars,array){

    }
};

////Bitwise functions over bitArrays

//Hamming distance
template <int n, typename T = uint64_t>
int HW_AND(std::vector<bitarray<T>*>& a, std::vector<int>& indices){
    T tmp;
    int hw = 0, ln = a[0]->size();
    for (int i = 0; i < ln; ++i) {

        //tmp = a[indices[0]]->operator[](i) & a[indices[1]]->operator[](i) & a[indices[2]]->operator[](i) & a[indices[2]]->operator[](i);
//        tmp = a[indices[0]]->operator[](i) & a[indices[1]]->operator[](i) & a[indices[2]]->operator[](i) & a[indices[3]]->operator[](i);
        tmp = a[indices[0]]->operator[](i);
        for (int j = 1; j < n; ++j) {
            tmp &= a[indices[j]]->operator[](i);
        }

        hw +=  popCount (tmp);
    }
    return hw;
}
//Hamming distance
template <typename T = uint64_t >
int HW_AND(bitarray<T>& a, bitarray<T>& b){
    if(a.getNumVars() != b.getNumVars() )
        std::cout << "diferent sizes";
    T tmp;
    int hw = 0;
    for (int i = 0; i < a.size() ; ++i) {
        tmp = a[i] & b[i];
        hw += popCount(tmp);
    }
    return hw;
}

template <typename T = uint64_t >
int HW_AND(bitarray<T>& a, bitarray<T>& b, bitarray<T>& c){
    if(a.getNumVars() != b.getNumVars() || c.getNumVars() != b.getNumVars() )
        std::cout << "diferent sizes";
    T tmp;
    int hw = 0;
    for (int i = 0; i < a.size() ; ++i) {
        tmp = a[i] & b[i] & c[i];
        hw += __builtin_popcountll (tmp);
    }
    return hw;
}


////////////////////////XOR////////////////////////
template <typename T = uint64_t >
bitarray<T>& XOR(bitarray<T>& result, bitarray<T>& a, bitarray<T>& b){
     if(a.getNumVars() != b.getNumVars() )
        std::cout << "diferent sizes";

    for (int i = 0; i < a.size() ; ++i) {
        result[i] = a[i] ^ b[i];
    }
    return result;
}
/*
bitarray& XOR(bitarray& a, bitarray& b){
    bitarray *result = new bitarray(a.getNumVars());
    return XOR(*result, a, b);
}
 */
////////////////////////AND////////////////////////
template <typename T = uint64_t >
bitarray<T>& AND(bitarray<T>& result, bitarray<T>& a, bitarray<T>& b){
    if(a.getNumVars() != b.getNumVars() )
        std::cout << "diferent sizes";

    for (int i = 0; i < a.size() ; ++i) {
        result[i] = a[i] & b[i];
    }
    return result;
}

/*
bitarray& AND(bitarray& a, bitarray& b){
    bitarray *result = new bitarray(a.getNumVars());
    return AND(*result, a, b);
}
*/



#endif //DYNAMIC_BITSET_TERM_H
