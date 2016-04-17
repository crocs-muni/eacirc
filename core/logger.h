#include "teestream.h"
#include <cassert>
#include <fstream>
#include <iostream>

struct Logger {
private:
    std::ofstream _file;
    Teestream _tee{_file, std::cout};

public:
    struct Datestamp {};
    struct Timestamp {};

    Logger(const std::string file) : _file(file) {
        assert(instance == nullptr);
        Logger::instance = this;
    }
    ~Logger() { Logger::instance = nullptr; }

    static std::ostream& warning() { return out() << " [warning] "; }
    static std::ostream& error() { return out() << " [error] "; }
    static std::ostream& info() { return out() << " [info] "; }

protected:
    static std::ostream& out();

private:
    static Logger* instance;
};

std::ostream& operator<<(std::ostream& os, Logger::Timestamp);
std::ostream& operator<<(std::ostream& os, Logger::Datestamp);
