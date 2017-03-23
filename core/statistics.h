#pragma once

#include <cstdint>
#include <vector>

namespace core {
namespace statistics {

struct Test {
  double critical_value;
  double test_statistic;
};

double two_sample_chisqr(const std::vector<std::uint64_t>& distribution_a,
                         const std::vector<std::uint64_t>& distribution_b);

double ks_uniformity_test(std::vector<double> samples);
double ks_uniformity_critical_value(std::size_t number_of_samples, unsigned significance_level);

} // namespace statistics
} // namespace core
