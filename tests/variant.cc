#include <catch.hpp>
#include <ea-variant.h>

using namespace ea;

TEST_CASE("variant") {
    SECTION("empty copy") {
        const variant<int, float> a;
        REQUIRE(a.empty());

        variant<int, float> b(a);
        REQUIRE(b.empty());

        b = a;
        REQUIRE(b.empty());
    }

    SECTION("empty move") {
        variant<int, float> a;
        REQUIRE(a.empty());

        variant<int, float> b(std::move(a));
        REQUIRE(b.empty());

        b = std::move(a);
        REQUIRE(b.empty());
    }

    SECTION("construct & assign") {
        variant<int, float> a(7);
        REQUIRE(!a.empty());
        REQUIRE(a.is<int>());
        REQUIRE(a.as<int>() == 7);

        variant<int, float> b(1);
        REQUIRE(b.as<int>() == 1);

        b = a;
        REQUIRE(b.as<int>() == 7);

        a = 6;
        REQUIRE(a.as<int>() == 6);

        a = 7.7f;
        REQUIRE(a.is<float>());
        REQUIRE(a.as<float>() == Approx(7.7f));

        b = a;
        REQUIRE(b.is<float>());
        REQUIRE(b.as<float>() == Approx(7.7f));
    }

    SECTION("complex") {
        variant<int, std::string, std::vector<int>> a;
        a = 7;
        a = std::string("hi");

        std::vector<int> vec{3, 2};
        a = vec;

        auto l = a.as<std::vector<int>>();

        REQUIRE(l == vec);

        a = 7;
    }
}
