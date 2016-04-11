#include "logger.h"
#include <iomanip>
#include <iostream>

std::ostream& Logger::date_stamp()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    return _tee << std::put_time(&tm, "%F");
}

std::ostream& Logger::time_stamp()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    return _tee << std::put_time(&tm, "%T");
}
