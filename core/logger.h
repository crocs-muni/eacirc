#pragma once

#include <ostream>

struct logger {
    static void warning(std::string msg) {
        logger::out() << "[warning] " << msg << std::endl;
    }

    static void error(std::string msg) {
        logger::out() << "[error] " << msg << std::endl;
    }

    static void info(std::string msg) {
        logger::out() << "[info] " << msg << std::endl;
    }

private:
    static std::ostream &out();
};
