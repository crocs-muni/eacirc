#include <catch.hpp>
#include <ea-iterators.h>
#include <vector>

using namespace ea;

TEST_CASE("step_iterator") {
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    SECTION("increment") {
        auto iterator = make_step_iterator(vec.begin(), 2);

        REQUIRE(*iterator == 1);
        ++iterator;
        REQUIRE(*iterator == 3);
    }
}
