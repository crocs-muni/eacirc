#include "logger.h"
#include <ctime>
#include <iostream>

std::ostream& logger::out() {
    return std::cout << logger::time() << " ";
}

std::string logger::date() {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    char str[100];
    std::strftime(str, sizeof(str), "%d %b %Y", &tm);
    return str;
}

std::string logger::time() {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    char str[100];
    std::strftime(str, sizeof(str), "%H:%M:%S", &tm);
    return str;
}
