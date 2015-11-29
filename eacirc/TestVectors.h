#pragma once

#include "Byte.h"
#include <memory>
#include "cxx_utils.h"

class TestVectors
{
public:
    TestVectors() = default;

    TestVectors( const size_t step, const size_t num )
        : step_( step )
        , num_( num )
        , data_( std::make_unique<Byte[]>( num * step ) )
    {}

    TestVectors& operator=(TestVectors&& rhs)
    {
        step_ = rhs.step_;
        num_ = rhs.num_;
        data_ = std::move( rhs.data_ );
        return *this;
    }
public:
    Byte* operator[]( size_t i ) { return &data_[i * step_]; }
    const Byte* operator[]( size_t i ) const { return &data_[i * step_]; }

    Byte* data() { return data_.get(); }
    const Byte* data() const { return data_.get(); }

    size_t step() const { return step_; }
    size_t num() const { return num_; }
private:
    size_t step_;
    size_t num_;
    std::unique_ptr<Byte[]> data_;
};
