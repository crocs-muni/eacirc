#include <exception>
#include <iostream>
#include <sstream>

#include "ResultProcessor.h"
#include "FileGenerator.h"

std::string writeUsage();
bool setProgramOptions(char * argv[] , int args , int & mode , int & pprocNum , std::string & path);
int strtoi(std::string s);

int main(int args , char * argv[]) {
    bool executed = false;
    bool validArguments = false;
    int mode = 0;
    int pprocNum = 0;
    std::string path;

    validArguments = setProgramOptions(argv, args, mode, pprocNum, path);
    oneclickLogger.setLogToConsole(true);

    //Generation of:
    //    -configuration files for EACirc
    //    -script for creating workunits on BOINC server
    //    -script for downloading results from BOINC server
    //Based on config file given in third argument.
    //FileGenerator * fg = NULL;
    if (validArguments && mode == MODE_FILE_GENERATION) {
        executed = true;
        FileGenerator * fg = NULL;
        try {
            fg = new FileGenerator(path);
            delete fg;
        }
        catch (std::runtime_error e) {
            oneclickLogger << FileLogger::LOG_ERROR << e.what() << "\n";
            delete fg;
        }
    }

    //Processing of files in result directory given in second argument.
    //Creates files with results.
    if (validArguments && mode == MODE_RESULT_PROCESSING) {
        executed = true;
        try {
            ResultProcessor rp = ResultProcessor(path, pprocNum);
        }
        catch (std::runtime_error e) {
            oneclickLogger << FileLogger::LOG_ERROR << e.what() << "\n";
        }
    }

    //Nothing happened
    if (!executed) {
        oneclickLogger << FileLogger::LOG_ERROR << "wrong usage of arguments\n";
        oneclickLogger << writeUsage();
    }

    return 0;
}

std::string writeUsage() {
    std::stringstream ss;
    ss << "[USAGE] Application takes these arguments: -mode pproc path\n";
    ss << " -mode : use \"-g\" (file generation) or \"-p\" (result processing)\n";
    ss << " pproc : defines post-processor that will be used in result processing mode (-p)\n";
    ss << "         use integer value (1 or 2). For more information refer to documentation\n";
    ss << "  path : path to config file (-g) or result directory (-p)\n";
    return ss.str();
}

bool setProgramOptions(char * argv[]  , int args , int & mode , int & pprocNum , std::string & path) {
    int next = 1;
    if(next == args) return false;

    //mode = strtoi(argv[next]);
    if(strcmp(argv[next] , "-g") == 0) mode = MODE_FILE_GENERATION;
    if(strcmp(argv[next] , "-p") == 0) mode = MODE_RESULT_PROCESSING;
    next++;
    if(next == args) return false;

    if(mode == MODE_RESULT_PROCESSING && args == 4) {
        pprocNum = strtoi(argv[next]);
        next++;
    } else {
        pprocNum = 1;
    }
    if(next == args) return false;

    path = argv[next];
    next++;
    if(next != args) return false;
    return true;
}

int strtoi(std::string s) {
    int x;
    try {
        x = std::stoi(s , nullptr);
        return x;
    } catch(std::invalid_argument e) {
        return 0;
    } catch(std::out_of_range e) {
        return 0;
    }
}
