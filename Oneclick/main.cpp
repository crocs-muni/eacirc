//almost funny header here

//Boost libraries Regex and Filesystem included now - 9.1.2015
//TODO
//	-EACirc dependency - refactor would be nice

#include <exception>
#include <iostream>
#include <sstream>

#include "ResultProcessor.h"
#include "FileGenerator.h"

std::string writeUsage();

int main(int args , char * argv[]) {
	bool executed = false;
	if(args < 3 || args > 4) {
		executed = true;
		oneclickLogger.setLogToConsole(true);
		oneclickLogger << FileLogger::LOG_ERROR << "missing arguments\n";
		oneclickLogger << writeUsage();
	}
	bool logToConsole = false;
	char * mode = NULL;
	char * path = NULL;

	if(args == 3) {
		mode = argv[1];
		path = argv[2];
	}

	if(args == 4) {
		if(strcmp(argv[1] , "-log2c") == 0) logToConsole = true;
		mode = argv[2];
		path = argv[3];
	}
	oneclickLogger.setLogToConsole(logToConsole);

	//Generation of:
	//	-configuration files for EACirc
	//	-script for creating workunits on BOINC server
	//	-script for downloading results from BOINC server
	//Based on config file given in second argument.
	if(!executed && strcmp(mode , "-g") == 0) {
		executed = true;
		try {
			FileGenerator fg = FileGenerator(path);
		} catch(std::runtime_error e) {
			oneclickLogger << FileLogger::LOG_ERROR << e.what() << "\n";
		}
	}

	//Processing of files in result directory given in second argument.
	//Creates files with results.
	if(!executed && strcmp(mode , "-p") == 0) {
		executed = true;

		try {
			ResultProcessor rp = ResultProcessor(path);
		} catch(std::runtime_error e) {
			oneclickLogger << FileLogger::LOG_ERROR << e.what() << "\n";
		}
	}
	
	//Nothing happened
	if(!executed) {
		oneclickLogger << FileLogger::LOG_ERROR << "unknown \"-mode\" argument\n";
		oneclickLogger << writeUsage();
	}

	return 0;
}

std::string writeUsage() {
	std::stringstream ss;
	ss << "[USAGE] Application takes three arguments: -log2c -mode path\n";
	ss << "-log2c : use only if you want to see log in console, leave empty otherwise\n";
	ss << " -mode : use \"-g\" (file generation) or \"-p\" (result processing)\n";
	ss << "  path : path to config file (-g) or result directory (-p)\n";
	return ss.str();
}