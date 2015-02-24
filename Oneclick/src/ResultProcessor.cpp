#include "ResultProcessor.h"

ResultProcessor::ResultProcessor(std::string path , int pprocNum) {
	fs::directory_iterator end;
	fs::directory_iterator dirIter(path);
	if(dirIter == end) throw std::runtime_error("given argument is not a path to existing directory: " + path);
	oneclickLogger << FileLogger::LOG_INFO << "started processing results\n\n";

	initPProcessor(pprocNum);

	std::string algName;
	std::string pValues;
	std::vector<std::string> configPaths;
	std::vector<std::string> logPaths;

	//pprocessor = new PValuePostPr();

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

				//Checks errors and warnings, if okay run PostProcessor
				oneclickLogger << FileLogger::LOG_INFO << "processing results, checking errors/warnings in logs\n";
				*dirLogger << FileLogger::LOG_INFO << "processing results, checking errors/warnings in logs\n";
				
				if(algName.length() == 0) algName = dirIter.name();
				pprocessor->setBatchDirectoryPath(dirIter.path() + "/");
				pprocessor->setBatchName(algName);
				checkErrorsProcess(logPaths , dirLogger);
				pprocessor->calculateBatchResults();

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

	pprocessor->saveResults();
	oneclickLogger << FileLogger::LOG_INFO << "finished processing results\n";
}

bool ResultProcessor::checkConfigs(std::vector<std::string> configPaths, std::string & algName , FileLogger * dirLogger) {
	bool isSampleSet = false;
	int badConfigCount = 0;
	std::string sampleConfig;
	std::string currentConfig;

    for(unsigned i = 0 ; i < configPaths.size() ; i++) {
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

void ResultProcessor::checkErrorsProcess(std::vector<std::string> logPaths , FileLogger * dirLogger) {
	int errorCount = 0;
	int wrnCount = 0;
	bool validity;

	std::regex errPatt   ("\\[\\d\\d:\\d\\d:\\d\\d\\] error: .*");
	std::regex wrnPatt   ("\\[\\d\\d:\\d\\d:\\d\\d\\] warning: .*");
	std::sregex_token_iterator endExpr;
	std::string logFile;
	std::string wuDirectory;
    for(unsigned i = 0 ; i < logPaths.size() ; i++) {
		errorCount = 0;
		wrnCount = 0;
		validity = true;
		logFile = Utils::readFileToString(logPaths[i]);
		wuDirectory = Utils::getPathWithoutLastItem(logPaths[i]);

		std::sregex_token_iterator errors(logFile.begin() , logFile.end() , errPatt , 0);
		std::sregex_token_iterator warnings(logFile.begin() , logFile.end() , wrnPatt , 0);

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

		if(errorCount == 0) {
			validity = pprocessor->process(wuDirectory);
			if(validity == false) {
				oneclickLogger << FileLogger::LOG_WARNING << "directory " << wuDirectory << " contains no valid results for processing\n";
				*dirLogger << FileLogger::LOG_WARNING << "directory " << wuDirectory << " contains no valid results for processing\n";
			}
		} else {
			*dirLogger << FileLogger::LOG_WARNING << Utils::getLastItemInPath(logPaths[i]) << " contains errors. Ignoring file.\n";
			oneclickLogger << FileLogger::LOG_WARNING << Utils::getLastItemInPath(logPaths[i]) << " contains errors. Ignoring file.\n";
		}

		if(errorCount != 0 || wrnCount != 0) {
			*dirLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
			oneclickLogger << FileLogger::LOG_INFO << Utils::getLastItemInPath(logPaths[i]) << " has " << Utils::itostr(wrnCount) << " warnings and " << Utils::itostr(errorCount) << " errors.\n";
		}

		logFile.erase();
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

		if(dirIter.is_file() && Utils::getFileIndex(dirIter.name()) == fileIndex) {
			paths.push_back(dirIter.path());
		}
	}
}

void ResultProcessor::initPProcessor(int pprocNum) {
	switch(pprocNum) {
	case PPROCESSOR_PVAL:
		pprocessor = new PValuePostPr();
		break;
	case PPROCESSOR_AVG:
		pprocessor = new AvgValPostPr();
		break;
	default:
		throw std::runtime_error("unknown post-processor set in command line arguments");
		break;
	}
}
