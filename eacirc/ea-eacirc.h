#pragma once

#include <random>
#include <string>

namespace ea {

struct eacirc {
    eacirc(const std::string config);

    void run();

private:
    std::size_t _num_of_epochs;
    std::size_t _epoch_duration;
    std::mt19937 _main_generator;
};

} // namespace ea
