#include "ConfigChecker.h"


ConfigChecker::ConfigChecker(std::string path) {
	if(!fs::exists(path) || !fs::is_directory(path)) throw runtime_error("given argument is not a path to existing directory: " + path);
	fs::path resultDir(path);

	fs::directory_iterator endIter;
	fs::directory_iterator dirIter(resultDir);
	std::string sampleConfig = readFileToString((dirIter->path()).generic_string());
	dirIter++;
	std::string currentConfig;

	for( ; dirIter != endIter ; dirIter++) {
		if(fs::is_regular_file(dirIter->status())) {
			if((dirIter->path()).generic_string().find(IDENTIFIER_CONFIG) != -1) {
				currentConfig = readFileToString((dirIter->path()).generic_string());
				if(sampleConfig.compare(currentConfig) != 0) {
					std::cout << "[WARNING] File " << (dirIter->path()).generic_string() << " differs from the rest of files in directory.\n";
				}
				currentConfig.erase();
			}
		}
	}
}