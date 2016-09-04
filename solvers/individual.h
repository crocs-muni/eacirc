#pragma once

#include <cstdint>
#include <utility>

namespace solvers {

    template <typename T, typename S> struct individual {
        T genotype;
        S score;

        individual()
            : genotype()
            , score() {}

        individual(T&& gen)
            : genotype(std::move(gen))
            , score() {}

        friend bool operator<(individual const& lhs, individual const& rhs) {
            return lhs.score < rhs.score;
        }

        friend bool operator<=(individual const& lhs, individual const& rhs) {
            return lhs.score <= rhs.score;
        }

        friend bool operator>(individual const& lhs, individual const& rhs) {
            return lhs.score > rhs.score;
        }

        friend bool operator>=(individual const& lhs, individual const& rhs) {
            return lhs.score >= rhs.score;
        }

        friend void swap(individual& lhs, individual& rhs) {
            using std::swap;
            swap(lhs.genotype, rhs.genotype);
            swap(lhs.score, lhs.score);
        }
    };

} // namespace solvers
