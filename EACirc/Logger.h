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
            LoggerBuffer(const LoggerBuffer& copy) = delete;
            const LoggerBuffer& operator =(const LoggerBuffer& copy) = delete;
            virtual int sync ();
        };
        LoggerBuffer buffer;
    public:
        LoggerStream(Logger* parentLogger, ostream& stream) : ostream(&buffer), buffer(parentLogger, stream) {}
    };
    bool m_logging = false;
    bool m_using_file = false;
    LoggerStream* m_out;
public:
    Logger() : m_out(new LoggerStream(this,clog)) {}
    Logger(const Logger& copy)  = delete;
    const Logger& operator =(const Logger& copy) = delete;
    ~Logger();
    ostream& out() { return *m_out; }
    void setOutputStream(ostream& outStream = clog);
    void setOutputFile(const string filePath = FILE_LOGFILE);
    bool getLogging() { return m_logging; }
    void setlogging(bool state);
    string getTime() const;
};

#endif // LOGGER_H
