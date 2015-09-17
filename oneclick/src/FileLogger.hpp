//Simple File Logger by mateandmetal
//Source: https://github.com/LaurentGomila/SFML/wiki/Source:-Simple-File-Logger-(by-mateandmetal)
//Modified

#ifndef FILELOGGER_HPP
#define FILELOGGER_HPP

#include <ctime>
#include <iostream>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <sstream>

#include "Version.h"

class FileLogger {

private:
    static std::string getTime() {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(buffer , 80 , "%H:%M:%S" , timeinfo);
        std::stringstream temp;
        temp << "[" << buffer << "] ";
        return temp.str();
    };

    static std::string getDate() {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(buffer , 80 , "%Y-%m-%d" , timeinfo);
        return std::string(buffer);
    }

public:

    enum e_logType { LOG_ERROR , LOG_WARNING , LOG_INFO };

    explicit FileLogger(std::string fname) : numWarnings(0U) , numErrors(0U) , toConsole(false) {
        myFile.open(fname);
        // Write the first lines
        if(myFile.is_open()) {
            myFile << "EACirc Oneclick application log (build " << GIT_COMMIT_SHORT << ")." << std::endl;
            myFile << "Date: " << getDate() << "\n\n";
            myFile << getTime() << "Logging started.\n";
        } else {
            throw std::runtime_error("can't open logger file: " + fname);
        }
    };

    ~FileLogger() {
        if(myFile.is_open()) {
            myFile << std::endl;

            // Report number of errors and warnings
            myFile << getTime() << numWarnings << " warnings" << std::endl;
            myFile << getTime() << numErrors << " errors" << std::endl;
            myFile << getTime() << "Logger shutdown.\n";
            myFile.close();
        }
        if(toConsole) {
            std::cout << getTime() << numWarnings << " warnings" << std::endl;
            std::cout << getTime() << numErrors << " errors" << std::endl;
            std::cout << getTime() << "Logger shutdown.\n";
        }
    }

    void setLogToConsole(bool c) {
        toConsole = c;
        if(toConsole) {
            std::cout << "EACirc Oneclick application log (build " << GIT_COMMIT_SHORT << ")." << std::endl;
            std::cout << "Date: " << getDate() << "\n\n";
            std::cout << getTime() << "Logging started.\n";
        }
    }


    // Overload << operator using log type
    friend FileLogger &operator << (FileLogger &logger , const e_logType l_type) {

        switch(l_type) {
        case FileLogger::e_logType::LOG_ERROR:
            logger.myFile << getTime() << "error: ";
            ++logger.numErrors;
            if(logger.toConsole) {
                std::cout << getTime() << "error: ";
            }
            break;

        case FileLogger::e_logType::LOG_WARNING:
            logger.myFile << getTime() << "warning: ";
            ++logger.numWarnings;
            if(logger.toConsole) {
                std::cout << getTime() << "warning: ";
            }
            break;

        default:
            logger.myFile << getTime() << "info: ";
            if(logger.toConsole) {
                std::cout << getTime() << "info: ";
            }
            break;
        } 

        return logger;
    }

    // Overload << operator using C style strings
    // No need for std::string objects here
    friend FileLogger &operator << (FileLogger &logger , const std::string & text) {
        logger.myFile << text;
        if(logger.toConsole) {
            std::cout << text;
        }
        return logger;
    }

    // Make it Non Copyable (or you can inherit from sf::NonCopyable if you want)
    FileLogger(const FileLogger &) = delete;
    FileLogger &operator= (const FileLogger &) = delete;

private:
    std::ofstream myFile;
    unsigned int numWarnings;
    unsigned int numErrors;
    bool toConsole;
};
#endif // FILELOGGER_HPP
