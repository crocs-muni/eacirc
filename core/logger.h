#pragma once

#include <ostream>

struct logger {
    static void warning(std::string msg) { logger::warning() << msg << std::endl; }
    static void error(std::string msg) { logger::error() << msg << std::endl; }
    static void info(std::string msg) { logger::info() << msg << std::endl; }

    static std::ostream& warning() { return logger::out() << "[warning] "; }
    static std::ostream& error() { return logger::out() << "[error] "; }
    static std::ostream& info() { return logger::out() << "[info] "; }

private:
    static std::ostream& out();
};
