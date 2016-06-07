#include "logger.h"
#include <iomanip>
#include <iostream>

Logger* Logger::instance = nullptr;

std::ostream& Logger::out() {
    assert(Logger::instance != nullptr);
    return Logger::instance->_tee << Timestamp{};
}

std::ostream& operator<<(std::ostream& os, Logger::Datestamp) {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    return os << std::put_time(&tm, "%F");
}

std::ostream& operator<<(std::ostream& os, Logger::Timestamp) {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    return os << std::put_time(&tm, "%T");
}
