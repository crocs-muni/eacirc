#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

struct two_sample_chisqr {
    two_sample_chisqr(std::size_t categories)
        : _histogram_a(categories)
        , _histogram_b(categories) {}

    template <typename Container> double operator()(Container const& a, Container const& b) {
        std::fill(_histogram_a.begin(), _histogram_a.end(), 0u);
        std::fill(_histogram_b.begin(), _histogram_b.end(), 0u);

        for (auto vec : a)
            for (std::uint8_t byte : vec)
                _histogram_a[byte % _histogram_a.size()]++;
        for (auto vec : b)
            for (std::uint8_t byte : vec)
                _histogram_b[byte % _histogram_b.size()]++;
        return _compute();
    }

private:
    std::vector<std::uint64_t> _histogram_a;
    std::vector<std::uint64_t> _histogram_b;

    double _compute() const;
};

struct ks_uniformity_test {
    ks_uniformity_test(std::vector<double> samples, unsigned significance_level)
        : critical_value(_compute_critical_value(samples.size(), significance_level))
        , test_statistic(_compute_uniformity_test(samples)) {}

    const double critical_value;
    const double test_statistic;

private:
    static double _compute_critical_value(std::size_t size, unsigned significance_level);
    static double _compute_uniformity_test(std::vector<double>& samples);
};
