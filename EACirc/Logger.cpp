#include "Logger.h"

// for documentation on using logger, see EACglobals.h

Logger::Logger() : m_logging(false), m_using_file(false),
	m_out(new LoggerStream(this,clog)) {}

Logger::~Logger() {
    out(LOGGER_INFO) << "Exiting." << endl;
    delete m_out;
}

void Logger::setOutputStream(ostream& outStream) {
    delete m_out;
    m_out =  new LoggerStream(this,outStream);
}

void Logger::setOutputFile(const string filePath) {
    std::remove(filePath.c_str());
    delete m_out;
    m_out = new LoggerStream(this,*(new ofstream(filePath, fstream::app)));
}

void Logger::setlogging(bool state) {
    if (state == m_logging) return;
    out() << "info: Logging disabled" << endl;
    m_logging = state;
    out() << "info: Logging enabled." << endl;
}

string Logger::getTime() const {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime (buffer,80,"%Y-%m-%d %H:%M:%S",timeinfo);
    stringstream temp;
    temp << "[" << buffer << "] ";
    return temp.str();
}

// When we sync the stream with the output.
// 1) Output time then the buffer
// 2) Reset the buffer
// 3) flush the actual output stream we are using.
int Logger::LoggerStream::LoggerBuffer::sync() {
    if (!m_parentLogger->getLogging()) {
        str("");
        return 0;
    }
    m_out << m_parentLogger->getTime() << str();
    str("");
    m_out.flush();
    return 0;
}

bool Logger::getLogging() {
    return m_logging;
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
