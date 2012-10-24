#include "Logger.h"

Logger::~Logger() {
    out() << "Exiting EACirc." << endl;
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
    m_logging = state;
    out() << (m_logging == true ? "Logging enabled." : "Logging disabled") << endl;
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
