#include "Logger.h"

Logger* eacircMainLogger;

void Logger::setOutputStream(ostream& outStream) {
    m_out = &outStream;
}

void Logger::setOutputFile(const string filePath) {
    std::remove(filePath.c_str());
    m_out = new ofstream(filePath, fstream::app);
}

Logger::~Logger() {
    insert("Exiting EACirc.");
    if (m_using_file) {
        delete m_out;
    }
}

void Logger::setlogging(bool state) {
    if (state == m_logging) return;
    m_logging = state;
    insert(m_logging == true ? "Logging enabled." : "Logging disabled");
}

string Logger::formatMessage(const string message) const {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime (buffer,80,"%Y-%m-%d %H:%M:%S",timeinfo);
    stringstream temp;
    temp << "[" << buffer << "] " << message;
    return temp.str();
}

void Logger::insert(const string message) {
    if (!m_logging) return;
    *m_out << formatMessage(message) << endl;
}
