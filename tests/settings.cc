#include <catch.hpp>
#include <ea-settings.h>

using namespace ea;

TEST_CASE("settings") {
    settings conf;

    conf = 5LL;
    REQUIRE(static_cast<long long>(conf) == 5);

    conf = std::string("ahoj");
    REQUIRE(static_cast<std::string>(conf) == "ahoj");

    REQUIRE(conf["hi"].empty());
}
