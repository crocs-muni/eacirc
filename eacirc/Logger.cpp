#include "Logger.h"
#include "Version.h"
#include "CommonFnc.h"
// for documentation on using logger, see EACglobals.h

Logger::Logger() : m_logfile(FILE_LOGFILE), m_out(NULL) {
    setOutputFile(FILE_LOGFILE);
}

Logger::~Logger() {
    out(LOGGER_INFO) << "Exiting logger." << endl;
    delete m_out;
}

void Logger::setOutputFile(const string filePath) {
    delete m_out;
    //remove old log
    removeFile(m_logfile.c_str());
    m_logfile = filePath;
    // remove new log
    removeFile(m_logfile.c_str());
    m_out = new LoggerStream(this,*(new ofstream(filePath, fstream::app)));
    outputBuildInfo();
}

void Logger::outputBuildInfo() {
    out(LOGGER_INFO) << "Logging enabled." << endl;
    out(LOGGER_INFO) << "EACirc framework (build " << GIT_COMMIT_SHORT << ")." << endl;
    out(LOGGER_INFO) << "current date: " << getDate() << endl;
}

string Logger::getTime() const {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime (buffer,80,"%H:%M:%S",timeinfo);
    stringstream temp;
    temp << "[" << buffer << "] ";
    return temp.str();
}

string Logger::getDate() const {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime (buffer,80,"%Y-%m-%d",timeinfo);
    return string(buffer);
}

// When we sync the stream with the output.
// 1) Output time then the buffer
// 2) flush the actual output stream we are using.
// 3) Reset the buffer
int Logger::LoggerStream::LoggerBuffer::sync() {
    // console output
    clog << m_parentLogger->getTime() << str();
    clog.flush();
    // file output
    m_file << m_parentLogger->getTime() << str();
    m_file.flush();
    // clear buffer
    str("");
    return 0;
}

ostream& Logger::out() {
    return *m_out;
}

ostream& Logger::out(int prefix) {
    switch (prefix) {
    case LOGGER_INFO:
        out() << "info: ";
        break;
    case LOGGER_WARNING:
        out() << "warning: ";
        break;
    case LOGGER_ERROR:
        out() << "error: ";
        break;
    default:
        out(LOGGER_WARNING) << "Unknown logger prefix." << endl;
        break;
    }
    return out();
}
