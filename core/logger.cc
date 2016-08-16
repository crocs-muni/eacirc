#include "logger.h"
#include <iostream>

std::ostream &logger::out() {
    return std::cout;
}
