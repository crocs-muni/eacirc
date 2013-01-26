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

class Logger {
private:
    //! helper class: LoggerStream just uses a LoggerBuffer
    class LoggerStream : public ostream {
        //! helper class: buffer that prefixes each line with current time
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

    //! is logging enabled?
    bool m_logging;
    //! is logging to fileset via logger interface?
    bool m_using_file;
    //! stream to send logs to
    LoggerStream* m_out;
public:
    Logger();
    // Logger(const Logger& copy)  = delete; //(not supprrted in MS VS)
    // const Logger& operator =(const Logger& copy) = delete; //(not supprrted in MS VS)
    ~Logger();

    /** makes stream for logs available
      * @return stream
      */
    ostream& out() { return *m_out; }

    /** set stream for logs
      * @param outStream    new stream for logging
      */
    void setOutputStream(ostream& outStream = clog);

    /** set logging into file (if existing, new logs are appended)
      * @param filePath     file for logs
      */
    void setOutputFile(const string filePath = FILE_LOGFILE);

    /** is logging enabled?
      * @return logging enabled?
      */
    bool getLogging() { return m_logging; }

    /** enable/disable logging
      * @param state
      */
    void setlogging(bool state);

    /** get current date and time formatted into string
      * @return time
      */
    string getTime() const;
};

#endif // LOGGER_H
