#pragma once

#include <core/memory.h>
#include <memory>

#include <core/dataset.h>
#include <core/json.h>

//#include <eacirc/circuit/circuit.h>
#include <eacirc/circuit/interpreter.h>

#include "../statistics.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace _impl {

    int count_true_bits(std::uint8_t x) {
#ifdef __GNUC__
        return __builtin_popcount(x);
#elif _MSC_VER
        return __popcnt16(x);
#else
#warning "CAUTION! Unsupported compiler! Slow trivial implementation is used"
        int count = 0;
        while (x) {
            if (x % 2)
                ++count;
            x >>= 1;
        }
        return count;
#endif
    }

} // namespace _impl

namespace evaluators {

    template <typename Circuit> struct evaluator {
        virtual ~evaluator() = default;

        virtual void change_datasets(dataset const& a, dataset const& b) = 0;

        virtual double apply(Circuit const& circuit) = 0;
    };

    template <typename Circuit> struct categories_evaluator : evaluator<Circuit> {
        categories_evaluator(std::size_t num_of_categories)
            : _chisqr(num_of_categories) {}

        void change_datasets(dataset const& a, dataset const& b) {
            _a = a;
            _b = b;
        }

        double apply(Circuit const& circuit) {
            circuit::interpreter<Circuit> kernel{circuit};

            _oa.clear();
            _ob.clear();

            std::transform(_a.begin(), _a.end(), std::back_inserter(_oa), kernel);
            std::transform(_b.begin(), _b.end(), std::back_inserter(_ob), kernel);

            return 1.0 - _chisqr(_oa, _ob);
        }

    private:
        dataset _a;
        dataset _b;
        std::vector<typename Circuit::output> _oa;
        std::vector<typename Circuit::output> _ob;
        two_sample_chisqr _chisqr;
    };

    template <typename Circuit> struct weight_evaluator : evaluator<Circuit> {
        weight_evaluator(bool normalise)
            : _normalise(normalise) {}

        void change_datasets(dataset const& a, dataset const& b) {
            _a = a;
            _b = b;
        }

        double apply(Circuit const& circuit) {
            circuit::interpreter<Circuit> kernel{circuit};

            _oa.clear();
            _ob.clear();

            std::transform(_a.begin(), _a.end(), std::back_inserter(_oa), kernel);
            std::transform(_b.begin(), _b.end(), std::back_inserter(_ob), kernel);

            std::size_t bits_count_a = _oa.size() * Circuit::output_size * 8;
            std::size_t bits_count_b = _oa.size() * Circuit::output_size * 8;

            //                          ones in a           zeros in b
            std::size_t oposite_guesses = count_ones(_oa) + (bits_count_b - count_ones(_ob));

            if (_normalise)
                return normalized_p_val(oposite_guesses, bits_count_a + bits_count_b); // pval
            else {
                std::size_t total = bits_count_a + bits_count_b; // <0,1>
                return double(std::max(oposite_guesses, total - oposite_guesses) - total / 2) /
                       double(total);
            }
        }

    private:
        std::size_t count_ones(const std::vector<typename Circuit::output>& outputs) {
            std::size_t count = 0;

            for (auto& out_vec : outputs) {
                for (auto byte : out_vec) {
                    count += _impl::count_true_bits(byte);
                }
            }
            return count;
        }

        double normalized_p_val(std::size_t oposite_guesses, std::size_t total) {
            // we do not prefer to find ones or zeros, take what is higher
            std::size_t max_guesses = std::max(oposite_guesses, total - oposite_guesses);

            double p = normal_estimate(max_guesses, 0.5, total);
            if (p < 0.5)
                return 0; // we do not want negativ fitness
            else
                return (p - 0.5) * 2;
        }

        double erf(double z) {
            double t = 1.0 / (1.0 + 0.5 * std::abs(z));
            // use Horner's method
            // clang-format off
            double ans = 1 - t * std::exp(-z * z - 1.26551223 +
                             t * (1.00002368 + t * (0.37409196 +
                             t * (0.09678418 + t * (-0.18628806 +
                             t * (0.27886807 + t * (-1.13520398 +
                             t * (1.48851587 + t * (-0.82215223 +
                             t * (0.17087277))))))))));
            // clang-format on
            if (z >= 0.0)
                return ans;
            else
                return -ans;
        }

        double normal_estimate(std::size_t guesses, double p, std::size_t total) {
            double u = total * p; // always count of TV*bitsize of output
            double o = std::sqrt(u * (1 - p));

            return 0.5 * (1 + erf((guesses - u) / (o * std::sqrt(2))));
        }

        dataset _a;
        dataset _b;
        std::vector<typename Circuit::output> _oa;
        std::vector<typename Circuit::output> _ob;
        bool _normalise;
    };

    template <typename Circuit>
    std::unique_ptr<evaluator<Circuit>> make_evaluator(const json& config) {

        std::string type = config.at("type");

        if (type == "categories-evaluator")
            return std::make_unique<categories_evaluator<Circuit>>(
                    std::size_t(config.at("num-of-categories")));
        if (type == "weight-evaluator")
            return std::make_unique<weight_evaluator<Circuit>>(bool(config.at("normalise")));

        throw std::runtime_error("requested block cipher named \"" + type +
                                 "\" is either broken or does not exists");
    }
}; // namespace evaluators
