#include "ResultProcessor.h"

ResultProcessor::ResultProcessor(std::string path) {
	fs::directory_iterator end;
	fs::directory_iterator dirIter(path);
	if(dirIter == end) throw std::runtime_error("given argument is not a path to existing directory: " + path);
	oneclickLogger << FileLogger::LOG_INFO << "started processing results\n\n";
	std::string algName;
	std::string pValues;
	std::vector<std::string> configPaths;
	std::vector<std::string> logPaths;

	for(; dirIter != end ; dirIter++) {
		if(dirIter.is_directory() && dirIter.name().compare(".") != 0 && dirIter.name().compare("..")) {
			//Creates directory logger. All directory specific info are logged here also.
			FileLogger * dirLogger = new FileLogger(dirIter.path() + "/batch.log");
			oneclickLogger << FileLogger::LOG_INFO << "processing batch: " << dirIter.name() << "\n";
			*dirLogger << FileLogger::LOG_INFO << "processing batch: " << dirIter.name() << "\n";

			//Getting paths to all logs and configs in subdirectories
			getFilePaths(dirIter.path() , configPaths , INDEX_CONFIG);
			getFilePaths(dirIter.path() , logPaths , INDEX_EACIRC);
			oneclickLogger << FileLogger::LOG_INFO << Utils::itostr(configPaths.size()) << " configs and " << Utils::itostr(logPaths.size()) << " logs in batch\n";
			*dirLogger << FileLogger::LOG_INFO << Utils::itostr(configPaths.size()) << " configs and " << Utils::itostr(logPaths.size()) << " logs in batch\n";

			//Check for config consistency, won't procceed otherwise
			oneclickLogger << FileLogger::LOG_INFO << "checking differences in configuration files\n";
			*dirLogger << FileLogger::LOG_INFO << "checking differences in configuration files\n";
			if(checkConfigs(configPaths , algName , dirLogger)) {
				//Getting scores and p-values from logs
				//Checks errors and warnings
				oneclickLogger << FileLogger::LOG_INFO << "getting score, checking errors/warnings in logs\n";
				*dirLogger << FileLogger::LOG_INFO << "getting score, checking errors/warnings in logs\n";
				Score dirScore;
				dirScore.setVal(checkErrorsGetScore(logPaths , pValues , dirLogger));
				if(algName.length() == 0) algName = dirIter.name();
				dirScore.setAlgName(algName);
				scores.push_back(dirScore);
				Utils::saveStringToFile(dirIter.path() + "/pValues.txt" , pValues);
				oneclickLogger << FileLogger::LOG_INFO << "batch processed\n\n";
				*dirLogger << FileLogger::LOG_INFO << "batch processed\n\n";
			} else {
				oneclickLogger << FileLogger::LOG_WARNING << "batch won't be processed. Remove invalid runs before processing!\n\n";
				*dirLogger << FileLogger::LOG_WARNING << "batch won't be processed. Remove invalid runs before processing!\n\n";
			}

			configPaths.clear();
			logPaths.clear();
			algName.clear();
			delete dirLogger;
		}
	}
	writeScores();
	oneclickLogger << FileLogger::LOG_INFO << "finished processing results\n";
}

bool ResultProcessor::checkConfigs(std::vector<std::string> configPaths, std::string & algName , FileLogger * dirLogger) {
	bool isSampleSet = false;
	int badConfigCount = 0;
	std::string sampleConfig;
	std::string currentConfig;

	for(int i = 0 ; i < configPaths.size() ; i++) {
		if(!isSampleSet) {
			sampleConfig = Utils::readFileToString(configPaths[i]);
			algName = getNotes(sampleConfig);
			isSampleSet = true;
		}
		currentConfig = Utils::readFileToString(configPaths[i]);
		if(sampleConfig.compare(currentConfig) != 0) {
			oneclickLogger << FileLogger::LOG_WARNING << "config " << Utils::getLastItemInPath(configPaths[i]) << " differs from the first config in batch\n";
			*dirLogger << FileLogger::LOG_WARNING << "config " << Utils::getLastItemInPath(configPaths[i]) << " differs from the first config in batch\n";
			badConfigCount++;
		}
	}

	if(badConfigCount == 0) {
		oneclickLogger << FileLogger::LOG_INFO << "no different configs in batch\n";
		*dirLogger << FileLogger::LOG_INFO << "no different configs in batch\n";
		return true;
	} else {
		oneclickLogger << FileLogger::LOG_WARNING << Utils::itostr(badConfigCount) << " different configs in batch\n";
		*dirLogger << FileLogger::LOG_WARNING << Utils::itostr(badConfigCount) << " different configs in batch\n";
		return false;
	}
}

float ResultProcessor::checkErrorsGetScore(std::vector<std::string> logPaths , std::string & pValues , FileLogger * dirLogger) {
	int errorCount = 0;
	int wrnCount = 0;
	int validFileCount = 0;
	int uniformFileCount = 0;
	bool uniformity;
	bool hasResult;
	bool validity;

	std::regex errPatt   ("\\[\\d\\d:\\d\\d:\\d\\d\\] error: .*");
	std::regex wrnPatt   ("\\[\\d\\d:\\d\\d:\\d\\d\\] warning: .*");
	std::regex uniPatt   ("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS is not in 5% interval -> is uniform\\.");
	std::regex nonUniPatt("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS is in 5% interval -> uniformity hypothesis rejected\\.");
	std::regex pValPatt  ("\\[\\d\\d:\\d\\d:\\d\\d\\] info:    KS Statistics: (.*)");

	std::smatch pVal;
	std::smatch emptyMatch;
	std::sregex_token_iterator endExpr;
	std::string logFile;

	for(int i = 0 ; i < logPaths.size() ; i++) {
		errorCount = 0;
		wrnCount = 0;
		uniformity = false;
		hasResult = false;
		validity = true;
		logFile = Utils::readFileToString(logPaths[i]);

		std::sregex_token_iterator errors(logFile.begin() , logFile.end() , errPatt , 0);
		std::sregex_token_iterator warnings(logFile.begin() , logFile.end() , wrnPatt , 0);
		std::sregex_token_iterator uniform(logFile.begin() , logFile.end() , uniPatt , 0);
		std::sregex_token_iterator nonUniform(logFile.begin() , logFile.end() , nonUniPatt , 0);
		std::regex_search(logFile , pVal , pValPatt);

		for(; errors != endExpr ; errors++) {
			errorCount++;
			*dirLogger << FileLogger::LOG_WARNING << "error in log file: " << Utils::getLastItemInPath(logPaths[i]) << " == " << *errors << "\n";
			oneclickLogger << FileLogger::LOG_WARNING << "error in log file: " << Utils::getLastItemInPath(logPaths[i]) << " == " << *errors << "\n";
		}

		for(; warnings != endExpr ; warnings++) {
			wrnCount++;
			*dirLogger << FileLogger::LOG_WARNING << "warning in log file: " << Utils::getLastItemInPath(logPaths[i]) << " == " << *warnings << "\n";
			oneclickLogger << FileLogger::LOG_WARNING << "warning in log file: " << Utils::getLastItemInPath(logPaths[i]) << " == " << *warnings << "\n";
		}

		if(uniform != endExpr && nonUniform == endExpr) { uniformity = true; hasResult = true; }
		if(uniform == endExpr && nonUniform != endExpr) { uniformity = false; hasResult = true; }
		if(uniform != endExpr && nonUniform != endExpr) { validity = false; }
		if(pVal.size() != 2) {hasResult = false; }

		if(validity) {
			if(errorCount == 0) {
				if(hasResult) {
					pValues.append(pVal[1]);
					pValues.append("\n");
					validFileCount++;
					if(uniformity) {
						uniformFileCount++;
					}
				} else {
					*dirLogger << FileLogger::LOG_WARNING << Utils::getLastItemInPath(logPaths[i]) << " contains no result. Ignoring file.\n";
					oneclickLogger << FileLogger::LOG_WARNING << Utils::getLastItemInPath(logPaths[i]) << " contains no result. Ignoring file.\n";
					*dirLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
					oneclickLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
				}
			} else {
				*dirLogger << FileLogger::LOG_WARNING << Utils::getLastItemInPath(logPaths[i]) << " contains errors. Ignoring file.\n";
				oneclickLogger << FileLogger::LOG_WARNING << Utils::getLastItemInPath(logPaths[i]) << " contains errors. Ignoring file.\n";
				*dirLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
				oneclickLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
			}

			if(errorCount == 0 && wrnCount != 0 && hasResult) {
				*dirLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
				oneclickLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
			}
		} else {
			*dirLogger << FileLogger::LOG_WARNING << Utils::getLastItemInPath(logPaths[i]) << " contains two or more inconsistent results. Ignoring file.\n";
			oneclickLogger << FileLogger::LOG_WARNING << Utils::getLastItemInPath(logPaths[i]) << "contains two or more inconsistent results. Ignoring file.\n";
			*dirLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
			oneclickLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
		}
		pVal = emptyMatch;
		logFile.erase();
	}

	if(validFileCount != 0) {
		return (float)uniformFileCount / (float)validFileCount;
	} else {
		return ERROR_NO_VALID_FILES;
	}
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
		result = stoi(splitted[0] , nullptr);
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

void ResultProcessor::getFilePaths(std::string directory , std::vector<std::string> & paths , int fileIndex) {
	fs::directory_iterator end;
	fs::directory_iterator dirIter(directory);
	if(dirIter == end) throw std::runtime_error("given argument is not a path to existing directory: " + directory);

	for(; dirIter != end ; dirIter++) {
		if(dirIter.is_directory() && dirIter.name().compare(".") != 0 && dirIter.name().compare("..") != 0) {
			getFilePaths(dirIter.path() , paths , fileIndex);
		}

		if(dirIter.is_file() && getFileIndex(dirIter.name()) == fileIndex) {
			paths.push_back(dirIter.path());
		}
	}
}