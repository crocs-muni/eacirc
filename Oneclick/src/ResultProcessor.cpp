#include "ResultProcessor.h"

ResultProcessor::ResultProcessor(std::string path) {
	fs::directory_iterator end;
	fs::directory_iterator dirIter(path);
	if(dirIter == end) throw std::runtime_error("given argument is not a path to existing directory: " + path);
	oneclickLogger << FileLogger::LOG_INFO << "started processing results\n";
	std::string algName;

	for(; dirIter != end ; dirIter++) {
		if(dirIter.is_directory() && dirIter.name().compare(".") != 0 && dirIter.name().compare("..")) {

			//Creates directory logger. All directory specific info are logged here also.
			FileLogger * dirLogger = new FileLogger(dirIter.path() + "/000_" + dirIter.name() + ".log");

			//Directory check
			if(checkConfigs(dirIter.path() , &algName , dirLogger)) {
				//Directory check unsuccesfull => no processing
				Score dirScore;
				dirScore.setVal(checkErrorsGetScore(dirIter.path() , dirLogger));
				if(algName.length() == 0) algName = dirIter.name();
				dirScore.setAlgName(algName);
				scores.push_back(dirScore);
			}
			algName.clear();
			delete dirLogger;
		}
	}
	writeScores();
	oneclickLogger << FileLogger::LOG_INFO << "finished processing results\n";
}

bool ResultProcessor::checkConfigs(std::string directory , std::string * algName , FileLogger * dirLogger) {
	fs::directory_iterator end;
	fs::directory_iterator dirIter(directory);
	if(dirIter == end) throw std::runtime_error("given argument is not a path to existing directory: " + directory);
	oneclickLogger << FileLogger::LOG_INFO << "checking differences in configs in directory: " << directory << "\n";
	*dirLogger << FileLogger::LOG_INFO << "checking differences in configs in directory: " << directory << "\n";

	bool isSampleSet = false;
	int configCount = 0;
	int logCount = 0;
	int badConfigCount = 0;
	std::string sampleConfig;
	std::string currentConfig;
	std::string filePath;

	for(; dirIter != end; dirIter++) {
		if(dirIter.is_file()) {

			switch(getFileIndex(dirIter.name())) {
				case INDEX_CONFIG:
					configCount++;
					filePath = dirIter.path();
					if(!isSampleSet) {
						sampleConfig = Utils::readFileToString(filePath);
						*algName = getNotes(sampleConfig);
						isSampleSet = true;
					}

					currentConfig = Utils::readFileToString(filePath);
					if(sampleConfig.compare(currentConfig) != 0) {
						oneclickLogger << FileLogger::LOG_WARNING << "config " << filePath << " differs from the first config in directory\n";
						*dirLogger << FileLogger::LOG_WARNING << "config " << filePath << " differs from the first config in directory\n";
						badConfigCount++;
					}

					filePath.erase();
					currentConfig.erase();
					break;
				case INDEX_LOG:
					logCount++;
					break;
			}
		}
	}

	oneclickLogger << FileLogger::LOG_INFO << Utils::itostr(configCount) << " configs and " << Utils::itostr(logCount) << " logs in directory\n";
	*dirLogger << FileLogger::LOG_INFO << Utils::itostr(configCount) << " configs and " << Utils::itostr(logCount) << " logs in directory\n";

	if(badConfigCount == 0) {
		oneclickLogger << FileLogger::LOG_INFO << "no different configs in directory\n\n";
		*dirLogger << FileLogger::LOG_INFO << "no different configs in directory\n\n";
		return true;
	} else {
		oneclickLogger << FileLogger::LOG_WARNING << Utils::itostr(badConfigCount) << " different configs in directory\n";
		oneclickLogger << FileLogger::LOG_WARNING << "directory " << directory << " won't be processed. Remove invalid runs before processing!\n\n";
		*dirLogger << FileLogger::LOG_WARNING << Utils::itostr(badConfigCount) << " different configs in directory\n";
		*dirLogger << FileLogger::LOG_WARNING << "directory " << directory << " won't be processed. Remove invalid runs before processing!\n\n";
		return false;
	}
	return false;
}

float ResultProcessor::checkErrorsGetScore(std::string directory , FileLogger * dirLogger) {
	fs::directory_iterator end;
	fs::directory_iterator dirIter(directory);
	if(dirIter == end) throw std::runtime_error("given argument is not a path to existing directory: " + directory);

	oneclickLogger << FileLogger::LOG_INFO << "getting score, checking errors/warnings in logs: " << directory << "\n\n";
	*dirLogger << FileLogger::LOG_INFO << "getting score, checking errors/warnings in logs: " << directory << "\n\n";

	int errorCount = 0;
	int wrnCount = 0;
	int validFileCount = 0;
	int uniformFileCount = 0;
	bool uniformity;
	bool hasResult;
	bool validity;
	std::string filePath;

	std::regex errPatt   ("\\[\\d\\d:\\d\\d:\\d\\d\\] error: .*");
	std::regex wrnPatt   ("\\[\\d\\d:\\d\\d:\\d\\d\\] warning: .*");
	std::regex uniPatt   ("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS is not in 5% interval -> is uniform\\.");
	std::regex nonUniPatt("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS is in 5% interval -> uniformity hypothesis rejected\\.");

	std::sregex_token_iterator endExpr;
	std::string logFile;

	for(; dirIter != end ; dirIter++) {
		if(dirIter.is_file() && getFileIndex(dirIter.name()) == INDEX_LOG) {
			filePath = dirIter.path();
			errorCount = 0;
			wrnCount = 0;
			uniformity = false;
			hasResult = false;
			validity = true;
			logFile = Utils::readFileToString(filePath);

			std::sregex_token_iterator errors(logFile.begin() , logFile.end() , errPatt , 0);
			std::sregex_token_iterator warnings(logFile.begin() , logFile.end() , wrnPatt , 0);
			std::sregex_token_iterator uniform(logFile.begin() , logFile.end() , uniPatt , 0);
			std::sregex_token_iterator nonUniform(logFile.begin() , logFile.end() , nonUniPatt , 0);

			for(; errors != endExpr ; errors++) {
				errorCount++;
				*dirLogger << FileLogger::LOG_WARNING << "error in log file: " << filePath << " == " << *errors << "\n";
				oneclickLogger << FileLogger::LOG_WARNING << "error in log file: " << filePath << " == " << *errors << "\n";
			}

			for(; warnings != endExpr ; warnings++) {
				wrnCount++;
				*dirLogger << FileLogger::LOG_WARNING << "warning in log file: " << filePath << " == " << *warnings << "\n";
				oneclickLogger << FileLogger::LOG_WARNING << "warning in log file: " << filePath << " == " << *warnings << "\n";
			}

			if(uniform != endExpr && nonUniform == endExpr) {
				uniformity = true;
				hasResult = true;
		    }

			if(uniform == endExpr && nonUniform != endExpr) {
				uniformity = false;
				hasResult = true;
			}

			if(uniform != endExpr && nonUniform != endExpr) {
				validity = false;
			}

			if(validity) {
				if(errorCount == 0) {
					if(hasResult) {
						validFileCount++;
						if(uniformity) {
							uniformFileCount++;
						}
					} else {
						*dirLogger << FileLogger::LOG_INFO << filePath << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
						oneclickLogger << FileLogger::LOG_INFO << filePath << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
						*dirLogger << FileLogger::LOG_WARNING << filePath << " contains no result. Ignoring file.\n\n";
						oneclickLogger << FileLogger::LOG_WARNING << filePath << " contains no result. Ignoring file.\n\n";
					}
				} else {
					*dirLogger << FileLogger::LOG_INFO << filePath << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
					oneclickLogger << FileLogger::LOG_INFO << filePath << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
					*dirLogger << FileLogger::LOG_WARNING << filePath << " contains errors. Ignoring file.\n\n";
					oneclickLogger << FileLogger::LOG_WARNING << filePath << " contains errors. Ignoring file.\n\n";
				}

				if(errorCount == 0 && wrnCount != 0 && hasResult) {
					*dirLogger << FileLogger::LOG_INFO << filePath << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n\n";
					oneclickLogger << FileLogger::LOG_INFO << filePath << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n\n";
				}
			} else {
				*dirLogger << FileLogger::LOG_INFO << filePath << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n\n";
				oneclickLogger << FileLogger::LOG_INFO << filePath << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n\n";
				*dirLogger << FileLogger::LOG_WARNING << filePath << " contains two or more inconsistent results. Ignoring file.\n\n";
				oneclickLogger << FileLogger::LOG_WARNING << filePath << "contains two or more inconsistent results. Ignoring file.\n\n";
			}
			logFile.erase();
		}
	}
	if(validFileCount != 0) {
		return (float)uniformFileCount / (float)validFileCount;
	} else {
		return ERROR_NO_VALID_FILES;
	}
	return 1.0;
}

void ResultProcessor::writeScores() {
	std::ofstream resultFile(FILE_PROCESSED_RESULTS , std::ios::out);
	if(!resultFile.is_open()) throw std::runtime_error("can't open file: " + (std::string)FILE_PROCESSED_RESULTS);

	for(int i = 0 ; i < scores.size() ; i++) {
		resultFile << scores[i].toString() << "\n";
	}

	resultFile.close();
}

int ResultProcessor::getFileIndex(std::string fileName) {
	int result = 0;
	std::vector<std::string> splitted = Utils::split(fileName , INDEX_SEPARATOR);
	
	try {
		result = stoi(splitted[splitted.size() - 1] , nullptr);
		return result;
	} catch (std::invalid_argument e) {
		return -1;
	} catch(std::out_of_range e) {
		return -1;
	}
}

std::string ResultProcessor::getNotes(std::string config) {
	std::regex notesPatt("<NOTES>(.*?)</NOTES>");

	std::smatch res;
	std::regex_search(config , res , notesPatt);

	if(res.size() != 2)
		return "";

	return res[1];
}

