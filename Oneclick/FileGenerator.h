#ifndef FILEGENERATOR_H
#define FILEGENERATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <sstream>
#include <fstream>

#include "XMLProcessor.h"
#include "ConfigParser.h"
#include "OneclickConstants.h"


class FileGenerator {
private:
	ConfigParser * parser;
public:
	/** Constructor for FileGenerator class. Inits ConfigParser class, then uses parsed
	  * info from XML file to generate config files for EACirc and perl scripts to upload
	  * and download them to BOINC server.
	  * @param path			path to XML config file for Oneclick.
	  */
	FileGenerator(std::string path);

	/** Destructor, deallocates ConfigParser class.
	  */
	~FileGenerator();
private:

	/** Generates XML configs specified in ConfigParser class, PERL script for uploading configs
	  * to BOINC server and script for downloading results.
	  * @throw runtime_error	if files can't be opened or written into
	  */
	void generateFiles();

	//ASAP
	void generateDownloadScript() { throw runtime_error("not implemented yet!"); };

	/** Writes beginning of perl script into file.
	  * @param script			opened file stream (doesn't close)
	  */
	void writeFirstUpScript(std::ofstream & script);
	/** Writes ending of perl script into file.
	  * @param script			opened file stream (doesn't close)
	  */
	void writeSecondUpScript(std::ofstream & script);
};

#endif //FILEGENERATOR_H