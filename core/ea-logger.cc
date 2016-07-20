#include "ea-logger.h"
#include <iomanip>

using namespace ea;

logger *logger::instance = nullptr;

std::ostream &logger::entry() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    return get()._tee << std::put_time(&tm, "%T ");
}
