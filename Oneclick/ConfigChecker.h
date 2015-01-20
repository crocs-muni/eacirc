#ifndef CONFIGCHECKER_H
#define CONFIGCHECKER_H

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include "XMLProcessor.h"
#include "OneclickConstants.h"

namespace fs = boost::filesystem;

//Checks config files in given directory
//Every file that has "config" in name.
//Takes first one as a sample and checks the rest - if they are different throw warning
class ConfigChecker {
	//attributes here
	//maybe logging differences into file?
	//i should make logger for everything!
	//great idea...
	std::string sampleConfig;
public:
	ConfigChecker(std::string path);
};


#endif //CONFIGCHECKER_H