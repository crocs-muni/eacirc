#pragma once

#include "individual.h"

#include <eacirc-core/dataset.h>
#include <eacirc-core/random.h>
#include <eacirc-core/view.h>

namespace solvers {

    struct solver {

        virtual ~solver() = default;

        virtual double run(std::uint64_t generations = 1) = 0;

        virtual double reevaluate(dataset const& a, dataset const& b) = 0;

        virtual view<std::vector<double>::const_iterator> scores() const  = 0;
    };
}
