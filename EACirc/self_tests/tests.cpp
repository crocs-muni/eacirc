#include "catch.hpp"

TEST_CASE("stupid/number equalities", "different numbers are not equal") {
    int number = 3;
    for (int i=1; i<5; i++) {
        CHECK(i !=number);
    }
}
