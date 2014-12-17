//HEADER TO BE ADDED

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "ConfigParser.h"
#include "FileGenerator.h"

int main(int args , char * argv[]) {
	if(args != 2) {
		std::cout << "[USAGE] One argument, path to config.xml\n";
	}

	try {
		FileGenerator * m = new FileGenerator(argv[1]);
	} catch(runtime_error e) {
		std::cerr << "[ERROR] " << e.what();
	}
	return 0;
}