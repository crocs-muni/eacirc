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

//! constants for different verbosity levels
//! MINIMAL: as few outputs as possible
#define LOGGER_VERBOSITY_MINIMAL 0
//! PRODUCTION: files stored from experiemnts
#define LOGGER_VERBOSITY_PRODUCTION 1
//! DEVELOPMENT: common development files
#define LOGGER_VERBOSITY_DEVELOPMENT 2
//! DEBUG: test vector debugging
#define LOGGER_VERBOSITY_DEBUG 3
//! DEEP_DEBUG: detailed info about test vector creation
#define LOGGER_VERBOSITY_DEEP_DEBUG 4

class Logger {
private:
    //! helper class: LoggerStream just uses a LoggerBuffer
    class LoggerStream : public ostream {
        //! helper class: buffer that prefixes each line with current time
        class LoggerBuffer : public stringbuf {
            ostream& m_file;
            Logger* m_parentLogger;
        public:
            LoggerBuffer(Logger* parentLogger, ostream& file) : m_file(file), m_parentLogger(parentLogger) {}
            virtual int sync ();
        };
        LoggerBuffer buffer;
    public:
        LoggerStream(Logger* parentLogger, ostream& stream) : ostream(&buffer), buffer(parentLogger, stream) {}
    };

    //! log file
    string m_logfile;
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

    /** output build info to log
      */
    void outputBuildInfo();

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

    /** set logging into file (if existing, new logs are appended)
      * @param filePath     file for logs
      */
    void setOutputFile(const string filePath = FILE_LOGFILE);
};

#endif // LOGGER_H
