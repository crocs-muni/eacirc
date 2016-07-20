#include <catch.hpp>
#include <ea-iterators.h>
#include <vector>

using namespace ea;

TEST_CASE("range_iterator") {
    std::vector<int> vec(100);

    SECTION("increment") {
        auto it = make_range_iterator(std::begin(vec), std::begin(vec) + 10);

        REQUIRE((*it).begin() == std::begin(vec));
        REQUIRE((*it).end() == std::begin(vec) + 10);

        ++it;

        REQUIRE((*it).begin() == std::begin(vec) + 1);
        REQUIRE((*it).end() == std::begin(vec) + 11);
    }

    SECTION("equality") {
        auto beg = make_range_iterator(std::begin(vec), 10);
        auto end = make_range_iterator(std::begin(vec), 10);

        REQUIRE(beg == end);
    }
}
