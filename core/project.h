#pragma once

#include "base.h"
#include <vector>

struct DataVectors {
    virtual ~DataVectors() = default;
};

struct TestedStream {
    virtual ~TestedStream() = default;

    virtual void read(DataVectors& tvs) = 0;
};

template <size_t S> class DataVecStorage : public DataVectors, public std::vector<DataVec<S>> {
public:
    DataVecStorage(size_t count) : std::vector<DataVec<S>>(count) {}
};

template <size_t In, size_t Out> struct Data {
    Data(size_t tv_count) : ins_A(tv_count), ins_B(tv_count), outs_A(tv_count), outs_B(tv_count) {}

    DataVecStorage<In> ins_A;
    DataVecStorage<In> ins_B;
    DataVecStorage<Out> outs_A;
    DataVecStorage<Out> outs_B;
};

template <unsigned I, unsigned O> class Evaluator {
protected:
    Data<I, O>* _data;

public:
    Evaluator() : _data(nullptr) {}
    virtual ~Evaluator() = default;

    void data(Data<I, O>* data) { _data = data; }
};
