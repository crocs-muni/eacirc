#pragma once

#include "variant.h"
#include <cstdint>
#include <pcg/pcg_random.hpp>
#include <random>
#include <utility>

template <typename Generator> struct seed_seq_from {
  using result_type = std::uint_least32_t;

  template <typename... Args>
  seed_seq_from(Args&&... args)
      : _rng(std::forward<Args>(args)...) {}

  seed_seq_from(seed_seq_from const&) = delete;
  seed_seq_from& operator=(seed_seq_from const&) = delete;

  template <typename I> void generate(I beg, I end) {
    for (; beg != end; ++beg)
      *beg = result_type(_rng());
  }

  constexpr std::size_t size() const {
    return (sizeof(typename Generator::result_type) > sizeof(result_type) &&
            Generator::max() > ~std::size_t(0UL))
               ? ~std::size_t(0UL)
               : size_t(Generator::max());
  }

private:
  Generator _rng;
};

struct polymorphic_generator {
  using result_type = std::uint8_t;

  template <typename Seeder> polymorphic_generator(const std::string& type, Seeder&& seeder) {
    if (type == "mt19937")
      _rng.emplace<std::mt19937>(std::forward<Seeder>(seeder));
    if (type == "pcg32")
      _rng.emplace<pcg32>(std::forward<Seeder>(seeder));
    else
      throw std::runtime_error("requested random generator named \"" + type +
                               "\" is not valid polymorphic generator");
  }

  static result_type min() { return std::numeric_limits<result_type>::min(); }
  static result_type max() { return std::numeric_limits<result_type>::max(); }

  result_type operator()() {
    switch (_rng.index()) {
    case decltype(_rng)::index_of<std::mt19937>():
      return std::uniform_int_distribution<result_type>()(_rng.as<std::mt19937>());
    case decltype(_rng)::index_of<pcg32>():
      return std::uniform_int_distribution<result_type>()(_rng.as<pcg32>());
    }

    throw std::logic_error("canot call polymorphic generator with undefined generator");
  }

private:
  core::variant<std::mt19937, pcg32> _rng;
};

using default_random_generator = pcg32;
using default_seed_source = seed_seq_from<default_random_generator>;
