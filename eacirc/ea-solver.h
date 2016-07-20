#pragma once

#include <ea-datastream.h>
#include <vector>

namespace ea {

struct solver {
    solver(std::vector<datastream> &streams)
        : _streams(streams) {}

    virtual std::vector<double> run() = 0;

protected:
    std::vector<datastream> &streams() { return _streams; }

private:
    std::vector<datastream> _streams;
};

} // namespace ea
