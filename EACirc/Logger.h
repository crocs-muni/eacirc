#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include "EACconstants.h"

using namespace std;

class Logger{
    bool m_logging = false;
    bool m_using_file = false;
    ostream* m_out = &clog;
    string formatMessage(const string message) const;
public:
    Logger() {}
    ~Logger();
    void setOutputStream(ostream& outStream = clog);
    void setOutputFile(const string filePath = FILE_LOGFILE);
    bool getLogging() { return m_logging; }
    void setlogging(bool state);
    void insert(const string message);
};

extern Logger* eacircMainLogger;

#endif // LOGGER_H
