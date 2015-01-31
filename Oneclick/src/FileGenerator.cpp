#include "FileGenerator.h"

FileGenerator::FileGenerator(std::string path) {
	parser = new ConfigParser(path);
	oneclickLogger << FileLogger::LOG_INFO << "started generation of scripts and files\n";
	generateFiles();
	oneclickLogger << FileLogger::LOG_INFO << "finished generation of scripts and files\n";
}

FileGenerator::~FileGenerator() {
	delete parser;
}

void FileGenerator::generateFiles() {
	std::vector<std::vector<int>> algorithmsRounds = parser->getAlgorithmsRounds();
	std::vector<int> numGenerations = parser->getNumGenerations();
	int project = parser->getProject();
	int clones = parser->getClones();
	std::string wuIdentifier = parser->getWuIdentifier();

	TiXmlNode * root = parser->getRoot();
	TiXmlNode * eacNode = NULL;
	eacNode = getXMLElement(root , PATH_EACIRC);

	//Creates directory, SHOULD be working on linux too
	Utils::createDirectory(DIRECTORY_CFGS);

	//Replacing keywords in scripts
	bool createWuFirstInsert = true;
	bool createDirectoryFirstInsert = true;
	bool downloadClonesFirstInsert = true;

	std::string uploadScriptSample =
		Utils::readFileToString((std::string)DIRECTORY_SCRIPT_SAMPLES + (std::string)FILE_SCRIPT_UPLOAD_SAMPLE);
	oneclickLogger << FileLogger::LOG_INFO << "file " << FILE_SCRIPT_UPLOAD_SAMPLE << " was loaded into memory\n";
	std::string createWuMethodPrototype = getMethodPrototype(uploadScriptSample , KEYWORD_METHOD_CREATE_WU);
	int uploadScriptPosition = uploadScriptSample.find(createWuMethodPrototype);
	replaceInString(&uploadScriptSample , KEYWORD_CLONES , Utils::itostr(clones));

	std::string downloadScriptSample = 
		Utils::readFileToString((std::string)DIRECTORY_SCRIPT_SAMPLES + (std::string)FILE_SCRIPT_DOWNLOAD_SAMPLE);
	oneclickLogger << FileLogger::LOG_INFO << "file " << FILE_SCRIPT_DOWNLOAD_SAMPLE << " was loaded into memory\n";
	std::string createDirectoryMethodPrototype = getMethodPrototype(downloadScriptSample , KEYWORD_METHOD_CREATE_DIRECTORY);
	std::string downloadClonesMethodPrototype = getMethodPrototype(downloadScriptSample , KEYWORD_METHOD_DOWNLOAD_CLONES);
	int downloadScriptPosition = min(downloadScriptSample.find(createDirectoryMethodPrototype) ,
		downloadScriptSample.find(downloadClonesMethodPrototype));
	replaceInString(&downloadScriptSample , KEYWORD_CLONES , Utils::itostr(clones));

	//Variables declaration
	TiXmlNode * n = NULL;
	std::string configName , wuName , notes , projectName;
	std::string algorithmName , wuDirectory , createWuMethod;
	std::string createDirectoryMethod , downloadClonesMethod;
	

	oneclickLogger << FileLogger::LOG_INFO << "started generating config files\n";
	for(int i = 0 ; i < numGenerations.size() ; i++) {
		for(int k = 0 ; k < algorithmsRounds.size() ; k++) {
			for(int l = 1 ; l < algorithmsRounds[k].size() ; l++) {

				//Tags in config file are set and human readable description of alg and project are given.
				OneclickConstants::setAlgorithmSpecifics(root , project , algorithmsRounds[k][0] , 
					algorithmsRounds[k][l] , &projectName , &algorithmName);
				notes = algorithmName;

				//Created names for workunit and config file
				if(wuIdentifier.length() > 0) wuName = wuIdentifier + "_";
				wuName = (wuName + Utils::getDate() + "_EAC_" + projectName + "_a" + Utils::itostr(algorithmsRounds[k][0]) +
					+ "r" + Utils::itostr(algorithmsRounds[k][l]) + "_g" + Utils::itostr(numGenerations[i]));

				configName = wuName;
				configName.append(".xml");

				//Adding line into upload script 
				createWuMethod = createWuMethodPrototype;
				replaceInString(&createWuMethod , KEYWORD_METHOD_CREATE_WU , DEFAULT_METHOD_CREATE_WU_NAME);
				replaceInString(&createWuMethod , KEYWORD_WU_NAME , wuName);
				replaceInString(&createWuMethod , KEYWORD_CONFIG_PATH , (std::string)DIRECTORY_CFGS + configName);
				uploadScriptPosition = insertIntoScript(&uploadScriptSample , createWuMethodPrototype , 
					createWuMethod , uploadScriptPosition , createWuFirstInsert);
				createWuFirstInsert = false;
				
				//Adding line into download script

				//Creation of directory for workunit results
				//Default directory name format is PROJECT_ALGORITHM_rRounds/
				//Have to end with separator
				wuDirectory = DIRECTORY_RESULTS + projectName + "_" + algorithmName + "_r" + Utils::itostr(algorithmsRounds[k][l]) + "/";
				createDirectoryMethod = createDirectoryMethodPrototype;
				replaceInString(&createDirectoryMethod , KEYWORD_METHOD_CREATE_DIRECTORY , DEFAULT_METHOD_CREATE_DIRECTORY_NAME);
				replaceInString(&createDirectoryMethod , KEYWORD_DIRECTORY_PATH , wuDirectory);
				downloadScriptPosition = insertIntoScript(&downloadScriptSample , createDirectoryMethodPrototype , 
					createDirectoryMethod , downloadScriptPosition , createDirectoryFirstInsert);
				createDirectoryFirstInsert = false;

				//Script for downloading workunit results
				downloadClonesMethod = downloadClonesMethodPrototype;
				replaceInString(&downloadClonesMethod , KEYWORD_METHOD_DOWNLOAD_CLONES , DEFAULT_METHOD_DOWNLOAD_CLONES_NAME);
				replaceInString(&downloadClonesMethod , KEYWORD_WU_NAME , wuName);
				replaceInString(&downloadClonesMethod , KEYWORD_WU_DIRECTORY , wuDirectory);
				downloadScriptPosition = insertIntoScript(&downloadScriptSample , downloadClonesMethodPrototype , 
					downloadClonesMethod , downloadScriptPosition , downloadClonesFirstInsert);
				downloadClonesFirstInsert = false;

				//Set tags in config file - human readable description and number of generations
				notes.append(" function with " + Utils::itostr(algorithmsRounds[k][l]) + " rounds.");
				if(setXMLElementValue(root , PATH_EAC_NOTES , notes) == STAT_INVALID_ARGUMETS) 
					throw std::runtime_error("invalid requested path in config: " + (std::string)PATH_EAC_NOTES);
				if(setXMLElementValue(root , PATH_EAC_GENS , Utils::itostr(numGenerations[i])) == STAT_INVALID_ARGUMETS)
					throw std::runtime_error("invalid requested path in config: " + (std::string)PATH_EAC_GENS);
				n = eacNode->Clone();
				
				//Saving XML config file
				if(saveXMLFile(n , DIRECTORY_CFGS + configName) != STAT_OK)
					throw std::runtime_error("can't save file: " + (std::string)DIRECTORY_CFGS + configName);

				//Cleaning up
				n = NULL;
				downloadClonesMethod.clear() ; wuDirectory.clear() ; createWuMethod.clear() ; createDirectoryMethod.clear();
				wuName.clear() ; notes.clear() ; configName.clear() ; algorithmName.clear() ; projectName.clear();
			}
		}
	}
	oneclickLogger << FileLogger::LOG_INFO << "finished generating config files\n";
	
	Utils::saveStringToFile(FILE_SCRIPT_UPLOAD , &uploadScriptSample);
	oneclickLogger << FileLogger::LOG_INFO << "created file " << FILE_SCRIPT_UPLOAD << "\n";
	Utils::saveStringToFile(FILE_SCRIPT_DOWNLOAD , &downloadScriptSample);
	oneclickLogger << FileLogger::LOG_INFO << "created file " << FILE_SCRIPT_DOWNLOAD << "\n";
}

std::string FileGenerator::getMethodPrototype(std::string source , std::string methodName) {
	int pos = source.find(methodName);
	if(pos == -1) throw std::runtime_error("can't find method name: " + methodName);
	std::string methodPrototype = source.substr(pos , source.find(DEFAULT_SCRIPT_LINE_SEPARATOR , pos) - pos + 1);
	return methodPrototype;
}

void FileGenerator::replaceInString(std::string * target , std::string replace , std::string instead) {
	int pos = target->find(replace);
	if(pos == -1) throw std::runtime_error("can't find string to be replaced: " + replace);
	target->replace(pos , replace.length() , instead);
}

int FileGenerator::insertIntoScript(std::string * target , std::string methodPrototype , std::string toInsert , int position , bool firstInsert) {
	if(/*position == target->find(methodPrototype)*/ firstInsert) {
		target->replace(position , methodPrototype.length() , toInsert);
		return (position + toInsert.length() + 2);
	} else {
		toInsert.push_back('\n');
		toInsert.push_back('\t');
		target->insert(position , toInsert);
		return (position + toInsert.length());
	}
}

