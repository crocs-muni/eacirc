//
// Created by syso on 16. 5. 2016.
//

#ifndef DYNAMIC_BITSET_DYNAMIC_BITSET_H
#define DYNAMIC_BITSET_DYNAMIC_BITSET_H

#include <inttypes.h>
#include <stdexcept>
#include <math.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>

typedef unsigned char u8;
typedef uint64_t u64;

template <typename T>
int TarraySize(int bitsize){
    return (bitsize + sizeof(T)*8-1) / (sizeof(T)*8);
}

template <typename T = uint64_t >
class dynamic_bitset{

public:

    dynamic_bitset<T>(int numVars = 0, void* array = NULL){
        if(array)setArray(numVars, array);
        else alloc(numVars);
    }
    void set(int bitIdx, bool bit = true){
        if (bitIdx >= _numVars){
            throw std::out_of_range("illegal bit position");
        }
		if(bit)
			_array[bitIdx / Tsize] |= (1ull << (bitIdx % Tsize));
		else{
			_array[bitIdx / Tsize] &= ~(1ull << (bitIdx % Tsize));
		}
    }
    bool get(int bitIdx){
        if (bitIdx >= _numVars){
            throw std::out_of_range("illegal bit position");
        }
        return (_array[bitIdx / Tsize] & (1ull  << (bitIdx % Tsize))) != 0;
    }
    void flip(int bitIdx){
        _array[bitIdx / Tsize] ^= (1 << (bitIdx % Tsize));
    }
    void alloc(int numVars){
        arraySize = TarraySize<T>(numVars);
        _array = new T[arraySize];
        reset();
        setNumVars(numVars);
    }
    void setNumVars(int numVars){
        _numVars = numVars;
    }
    int getNumVars(){
        return _numVars;
    }
    T& operator[](int idx){
        return _array[idx];
    }
    void setArray(int numVars, void* array){
        _array = (T*)array;
        arraySize = TarraySize<T>(numVars);
        setNumVars(numVars);
    }
    int size(){
        return arraySize;
    }
    void print(){
        for (int i = 0; i < _numVars ; ++i) {
            std::cout << get(i);
            if(i % 8 == 7) std::cout << " ";
        }
        std::cout << std::endl;
    }
    void reset(){
        for (int i = 0; i < arraySize; ++i) {
            _array[i] = 0;
        }
       // memset(_array,arraySize* Tsize,0);
    }
    void clear(){
        delete[](_array);
    }
    void rnd(){
        for (int i = 0; i < _numVars; ++i) {
            if(rand()%2 == 1) set(i);
        }
    }
    ~dynamic_bitset(){
        clear();
    }

private:
    T* _array;
    int _numVars;
    int arraySize;
    static const int Tsize = sizeof(T)*8;
};

#endif //DYNAMIC_BITSET_DYNAMIC_BITSET_H
