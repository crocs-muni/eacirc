//HEADER TO BE ADDED

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "OneclickConstants.h"
#include "ConfigParser.h"
#include "FileGenerator.h"

int main(int args , char * argv[]) {
	if(args != 2) {
		std::cout << "[USAGE] One argument, path to configuration XML file.\n";
		return 1;
	}

	try {
		FileGenerator * m = new FileGenerator(argv[1]);
		delete m;
	} catch(runtime_error e) {
		std::cerr << "[ERROR] " << e.what() << std::endl;
	}
	return 0;
}