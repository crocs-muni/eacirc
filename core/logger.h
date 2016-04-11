#include "teestream.h"
#include <fstream>
#include <iostream>

struct Logger {
private:
    std::ofstream _file{"bogo.log"};
    Teestream _tee{_file, std::cout};

public:
    ~Logger() { _tee << "exiting..." << std::endl; }

    static std::ostream& warning() { return out() << " [warning] "; }
    static std::ostream& error() { return out() << " [error] "; }
    static std::ostream& info() { return out() << " [info] "; }
    static std::ostream& out()
    {
        static Logger logger;
        return logger._tee;
    }

protected:
    std::ostream& date_stamp();
    std::ostream& time_stamp();
};
