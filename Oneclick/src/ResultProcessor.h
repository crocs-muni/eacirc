#ifndef RESULTPROCESSOR_H
#define RESULTPROCESSOR_H

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <regex>
#include <iostream>
#include <iomanip>

#include "FileSystem.h"
#include "OneclickConstants.h"

/** Class for storing result of single batch.
  */
class Score {
private:
	std::string algName;
	float val;
public:
	/** Sets attribute algName.
	  * @param a	will be algName
	  */
	void setAlgName(std::string a) {algName = a;};

	/** Sets attribute val.
	  * @param s	will be val
	  */
	void setVal(float s) {val = s;};

	/** Returns formatted string, contains algname    val.
	  * @return formatted result
	  */
	std::string toString() {
		std::stringstream result;
		result << std::setw(30);
		result << std::left;
		result << algName;

		result << std::setprecision(6);
		result << std::setw(15);
		result << std::right;
		if(val == ERROR_NO_VALID_FILES) {
			result << "no valid logs";
		} else {
			result << val;
		}
		return result.str();
	};
};

class ResultProcessor {
private:
	std::vector<Score> scores;
public:
	/** Constructor for ResultProcessor class.
	  * Checks given directory for config and log errors.
	  * If directory is valid, processes its results and writes them into file.
	  * @paran path				path to directory with results
	  *							(one directory per batch)
	  * @throw runtime_error	if invalid directory was given
	  */
	ResultProcessor(std::string path);
private:
	/** Checks configs in given directories. If some configs differ, directory
	  * is ignored in result processing.
	  * @param directory		directory to be checked
	  * @param algName			empty string, algorithm name will be set into it
	  * @param dirLogger		directory specific logger, logs
	  *							different configs
	  * @return					true if all configs are same, false otherwise
	  * @throw runtime_error	if invalid directory is given
	  */
	bool checkConfigs(std::string directory , std::string * algName , FileLogger * dirLogger);

	/** Check all logs in given directory for errors and warnings,
	  *	if log has no errors, result from log is processed.
	  * @param directory		dir to process
	  * @dirLogger				logs directory specifis events =
	  *							= errors, warnings, missing results, 
	  *							inconsistent results
	  * @return result			0-1 => uniformLogs/validLogs, if validLogs = 0
	  *							returns error constant (ERROR_NO_VALID_FILES)
	  * @throw runtime_error    invalid directory was given
	  */
	float checkErrorsGetScore(std::string directory , FileLogger * dirLogger);

	/** Writes scores for all batches into .txt file FILE_PROCESSED_RESULTS
	  * @throws runtime_error	when output file can't be opened
	  */
	void writeScores();

	/** Parses filename and gets index at the end.
	  * Index is separated by "_". Returns -1 if no
	  * index is found
	  * @param fileName			name of the file
	  * @return index
	  */
	int getFileIndex(std::string fileName);

	/** Retrieve tag <NOTES> from config file.
	  * @param config			config file loaded into string
	  * @return					string with notes
	  */
	std::string getNotes(std::string config);
};

#endif //RESULTPROCESSOR_H
