#include "eacirc.h"
#include <core/logger.h>
#include <stdexcept>
#include <system_error>

void test_environment();

int main() {
    Logger logger{"eacirc.log"};

    try {
        test_environment();

        Eacirc app("config.json");
        app.run();
    } catch (std::system_error& e) {
        Logger::error() << "System error " << e.code() << ": " << e.what()
                        << std::endl;
        return e.code().value();
    } catch (std::exception& e) {
        Logger::error() << e.what() << std::endl;
        return 1;
    } catch (...) {
        Logger::error() << "Unknown exception" << std::endl;
        return 1;
    }

    return 0;
}

#include <core/base.h>
#include <limits>

void test_environment() {
    if (std::numeric_limits<u8>::max() != 255)
        throw std::range_error("Maximum for unsigned char is not 255");
    if (std::numeric_limits<u8>::digits != 8)
        throw std::range_error("Unsigned char does not have 8 bits");
}
