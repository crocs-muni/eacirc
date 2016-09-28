//
// Created by syso on 16. 5. 2016.
//

#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#include "dynamic_bitset.h"
#include "bithacks.h"
#include "base.h"
#include <vector>

template <typename T = uint64_t >
class bitarray : public dynamic_bitset<T>{
public:
    bitarray(int numVars = 0, void* array = NULL):dynamic_bitset<T>(numVars,array){

    }
};

////Bitwise functions over bitArrays
template <typename T = uint64_t>
hwres HW(bitarray<T>& res){
    hwres hw = 0;
    int ln = res.size();
    for (int i = 0; i < ln; ++i) {
        hw += popCount(res[i]);
    }
    return hw;
}

//Hamming distance, evaluating on a term - second parameter
template <int n, typename T = uint64_t>
hwres HW_AND(std::vector<bitarray<T>*>& a, std::vector<int>& indices){
    T tmp;
    hwres hw = 0;
    int ln = a[0]->size();
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

//Hamming distance, without template parameter on degree
template <typename T = uint64_t>
hwres HW_AND(std::vector<bitarray<T>*>& a, std::vector<int>& indices){
    T tmp;
    hwres hw = 0;
    int ln = a[0]->size(), lnIdx = (int)indices.size();
    for (int i = 0; i < ln; ++i) {
        tmp = a[indices[0]]->operator[](i);
        for (int j = 1; j < lnIdx; ++j) {
            tmp &= a[indices[j]]->operator[](i);
        }

        hw +=  popCount (tmp);
    }
    return hw;
}

// res = term_eval(indices, a)
//   where a is a basis. i.e., a[0] = results for x_0, ...
// returns hw(res)
template <typename T = uint64_t>
hwres HW_AND(bitarray<T>& res, std::vector<bitarray<T>*>& a, std::vector<int>& indices){
    T tmp;
    hwres hw = 0;
    int ln = a[0]->size(), lnIdx = (int)indices.size();
    if (res.size() != ln){
        std::cout << "different sizes";
    }

    for (int i = 0; i < ln; ++i) {
        tmp = a[indices[0]]->operator[](i);

        for (int j = 1; j < lnIdx; ++j) {
            tmp &= a[indices[j]]->operator[](i);
        }

        res[i] = tmp;
        hw +=  popCount (tmp);
    }
    return hw;
}

//Hamming distance
template <typename T = uint64_t >
hwres HW_AND(bitarray<T>& a, bitarray<T>& b){
    if(a.getNumVars() != b.getNumVars() )
        std::cout << "different sizes";
    T tmp;
    hwres hw = 0;
    for (int i = 0; i < a.size() ; ++i) {
        tmp = a[i] & b[i];
        hw += popCount(tmp);
    }
    return hw;
}

template <typename T = uint64_t >
hwres HW_AND(bitarray<T>& res, bitarray<T>& a, bitarray<T>& b){
    if(a.getNumVars() != b.getNumVars() )
        std::cout << "different sizes";
    T tmp;
    hwres hw = 0;
    for (int i = 0; i < a.size() ; ++i) {
        tmp = a[i] & b[i];
        hw += popCount(tmp);
        res[i] = tmp;
    }
    return hw;
}

template <typename T = uint64_t >
hwres HW_AND3(bitarray<T>& a, bitarray<T>& b, bitarray<T>& c){
    if(a.getNumVars() != b.getNumVars() || c.getNumVars() != b.getNumVars() )
        std::cout << "different sizes";
    T tmp;
    hwres hw = 0;
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
        std::cout << "different sizes";

    for (int i = 0; i < a.size() ; ++i) {
        result[i] = a[i] ^ b[i];
    }
    return result;
}

// result = a xor b, returns hamming weight of the result
template <typename T = uint64_t >
hwres HW_XOR(bitarray<T>& result, bitarray<T>& a, bitarray<T>& b){
     if(a.getNumVars() != b.getNumVars() )
        std::cout << "different sizes";

    T tmp;
    hwres hw = 0;
    for (int i = 0; i < a.size() ; ++i) {
        tmp = a[i] ^ b[i];
        result[i] = tmp;
        hw += popCount(tmp);
    }

    return hw;
}

// returns hamming weight of the a xor b
template <typename T = uint64_t >
hwres HW_XOR(bitarray<T>& a, bitarray<T>& b){
     if(a.getNumVars() != b.getNumVars() )
        std::cout << "different sizes";

    T tmp;
    hwres hw = 0;
    for (int i = 0; i < a.size() ; ++i) {
        tmp = a[i] ^ b[i];
        hw += popCount(tmp);
    }

    return hw;
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
        std::cout << "different sizes";

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
