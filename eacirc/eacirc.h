#pragma once

#include "solvers/solver.h"
#include <core/project.h>
#include <memory>
#include <string>
#include <vector>

class Eacirc {
    std::unique_ptr<TestedStream> _stream_A;
    std::unique_ptr<TestedStream> _stream_B;
    std::unique_ptr<solvers::Solver> _solver;

    unsigned _num_of_tvs;
    unsigned _num_of_epochs;
    unsigned _change_frequency;
    unsigned _significance_level;

public:
    Eacirc(const std::string);
    ~Eacirc();

    void run();

protected:
    void ks_test_finish(std::vector<double>&) const;
};
