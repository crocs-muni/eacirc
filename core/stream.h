#pragma once

#include "dataset.h"
#include <algorithm>
#include <limits>
#include <vector>

struct counter {
    using value_type = std::uint8_t;
    using pointer = typename std::add_pointer<value_type>::type;
    using const_pointer = typename std::add_pointer<const value_type>::type;
    using iterator = std::vector<value_type>::iterator;
    using const_iterator = std::vector<value_type>::const_iterator;

    counter(std::size_t size)
        : _data(size) {
        std::fill(_data.begin(), _data.end(), std::numeric_limits<value_type>::min());
    }

    void increment() {
        for (value_type& value : _data) {
            if (value != std::numeric_limits<value_type>::max()) {
                ++value;
                break;
            }
            value = std::numeric_limits<value_type>::min();
        }
    }

    iterator begin() { return _data.begin(); }
    const_iterator begin() const { return _data.begin(); }

    iterator end() { return _data.end(); }
    const_iterator end() const { return _data.end(); }

    pointer data() { return _data.data(); }
    const_pointer data() const { return _data.data(); }

    std::size_t size() const { return _data.size(); }

private:
    std::vector<value_type> _data;
};

struct stream {
    using value_type = std::uint8_t;

    virtual ~stream() = default;

    virtual void read_dataset(dataset& set) = 0;
    virtual void read_raw(std::basic_ostream<value_type>& os, std::size_t size) = 0;
};

struct block_stream : stream {
    block_stream(std::size_t block)
        : _block(block) {}

    void read_dataset(dataset& set) override {
        for (auto& vec : set) {
            auto beg = vec.begin();
            auto end = vec.end();

            while (beg != end) {
                auto n = std::min(_block.size(), std::size_t(std::distance(beg, end)));

                generate();
                beg = std::copy_n(_block.data(), n, beg);
            }
        }
    }

    void read_raw(std::basic_ostream<value_type>& os, std::size_t size) override {
        while (size != 0) {
            auto n = std::min(_block.size(), size);

            generate();
            os.write(_block.data(), std::streamsize(n));
            if (!os.good())
                throw std::runtime_error("an error has occured during raw read");
            size -= n;
        }
    }

protected:
    virtual void generate() = 0;

    std::vector<value_type>& block() { return _block; }
    const std::vector<value_type>& block() const { return _block; }

private:
    std::vector<value_type> _block;
};
