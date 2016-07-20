#include <catch.hpp>
#include <ea-iterators.h>
#include <vector>

using namespace ea;

TEST_CASE("range") {
    SECTION("l-value construction") {
        std::vector<int> vec(10);

        auto range = make_range(vec);
        REQUIRE(range.begin() == vec.begin());
        REQUIRE(range.end() == vec.end());
    }

    SECTION("r-value construction") {
        struct Bar {
            Bar() = default;
            Bar(const Bar &) = delete;
            Bar(Bar &&) = delete;
        };

        std::vector<Bar> vec(10);

        auto range = make_range(std::move(vec));
        REQUIRE(range.begin() == vec.begin());
        REQUIRE(range.end() == vec.end());
    }
}
