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
    // LoggerStream just uses a LoggerBuffer
    class LoggerStream : public ostream {
        // buffer that prefixes each line with current time
        class LoggerBuffer : public stringbuf {
            ostream& m_out;
            Logger* m_parentLogger;
        public:
            LoggerBuffer(Logger* parentLogger, ostream& out) : m_out(out), m_parentLogger(parentLogger) {}
            // LoggerBuffer(const LoggerBuffer& copy) = delete; //(not supprrted in MS VS)
            // const LoggerBuffer& operator =(const LoggerBuffer& copy) = delete; //(not supprrted in MS VS)
            virtual int sync ();
        };
        LoggerBuffer buffer;
    public:
        LoggerStream(Logger* parentLogger, ostream& stream) : ostream(&buffer), buffer(parentLogger, stream) {}
    };
    bool m_logging;
    bool m_using_file;
    LoggerStream* m_out;
public:
    Logger();
    // Logger(const Logger& copy)  = delete; //(not supprrted in MS VS)
    // const Logger& operator =(const Logger& copy) = delete; //(not supprrted in MS VS)
    ~Logger();
    ostream& out() { return *m_out; }
    void setOutputStream(ostream& outStream = clog);
    void setOutputFile(const string filePath = FILE_LOGFILE);
    bool getLogging() { return m_logging; }
    void setlogging(bool state);
    string getTime() const;
};

#endif // LOGGER_H
