#include "eacirc.h"
#include <limits>

void test_environment();

int main() try {
    test_environment();

    eacirc app("settings.json");
    app.run();

    return 0;
} catch (std::exception &e) {
    // TODO
}

void test_environment() {
    if (std::numeric_limits<std::uint8_t>::max() != 255)
        throw std::range_error("Maximum for unsigned char is not 255");
    if (std::numeric_limits<std::uint8_t>::digits != 8)
        throw std::range_error("Unsigned char does not have 8 bits");
}
