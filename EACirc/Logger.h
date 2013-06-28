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

//! constants for logger output prefixes, use with logger.out(prefix)
//! INFO: operation successfull, just informing the user
#define LOGGER_INFO 0
//! WARNING: operation error, but recoverable - possibly something, that just may imply some error somewhere
#define LOGGER_WARNING 1
//! ERROR: operation error, unrecoverable - program should end without finishing computation
#define LOGGER_ERROR 2

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

    /** get current time formatted into string
      * @return time
      */
    string getTime() const;

    /**
     * get current date formatted into string
     * @return date
     */
    string getDate() const;

public:
    Logger();
    ~Logger();

    /** makes stream for logs available
      * @return stream
      */
    ostream& out();

    /** makes stream for logs available with common prefix
      * @param prefix       prefix constant
      * @return stream
      */
    ostream& out(int prefix);

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
    bool getLogging();

    /** enable/disable logging
      * @param state
      */
    void setlogging(bool state);
};

#endif // LOGGER_H
