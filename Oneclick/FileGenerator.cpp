#include "FileGenerator.h"

FileGenerator::FileGenerator(std::string path) {
	parser = new ConfigParser(path);
	generateFiles();
}

FileGenerator::~FileGenerator() {
	delete parser;
}

void FileGenerator::generateFiles() {
	std::vector<vector<int>> algorithmsRounds = parser->getAlgorithmsRounds();
	std::vector<int> numGenerations = parser->getNumGenerations();
	int project = parser->getProject();
	int clones = parser->getClones();
	std::string wuIdentifier = parser->getWuIdentifier();

	TiXmlNode * root = parser->getRoot();
	TiXmlNode * eacNode = NULL;
	eacNode = getXMLElement(root , PATH_EACIRC);
	//Changing keywords in scripts
	std::string uploadScriptSample = readFileToString((std::string)DIRECTORY_SCRIPT_SAMPLES + (std::string)FILE_SCRIPT_UPLOAD_SAMPLE);
	std::string createWuMethodPrototype = getMethodPrototype(uploadScriptSample , KEYWORD_METHOD_CREATE_WU);
	int uploadScriptPosition = uploadScriptSample.find(createWuMethodPrototype);
	replaceInString(&uploadScriptSample , KEYWORD_CLONES , itostr(clones));
	replaceInString(&uploadScriptSample , KEYWORD_DELAY_BOUND , itostr(parser->getDelayBound()));

	std::string downloadScriptSample = readFileToString((std::string)DIRECTORY_SCRIPT_SAMPLES + (std::string)FILE_SCRIPT_DOWNLOAD_SAMPLE);
	std::string createDirectoryMethodPrototype = getMethodPrototype(downloadScriptSample , KEYWORD_METHOD_CREATE_DIRECTORY);
	std::string downloadClonesMethodPrototype = getMethodPrototype(downloadScriptSample , KEYWORD_METHOD_DOWNLOAD_CLONES);
	int downloadScriptPosition = min(downloadScriptSample.find(createDirectoryMethodPrototype) , downloadScriptSample.find(downloadClonesMethodPrototype));
	replaceInString(&downloadScriptSample , KEYWORD_CLONES , itostr(clones));

	for(int i = 0 ; i < numGenerations.size() ; i++) {
		for(int k = 0 ; k < algorithmsRounds.size() ; k++) {
			for(int l = 1 ; l < algorithmsRounds[k].size() ; l++) {
				TiXmlNode * n = NULL;
				std::string configName;
				std::string wuName;
				std::string notes;
				std::string projectName;
				std::string algorithmName;
				std::string wuDirectory;
				std::string createWuMethod;
				std::string createDirectoryMethod;
				std::string downloadClonesMethod;

				//Tags in config file are set and human readable description of alg and project are given.
				setAlgorithmSpecifics(root , project , algorithmsRounds[k][0] , algorithmsRounds[k][l] , &projectName , &algorithmName);
				notes = algorithmName;

				//Created names for workunit and config file
				wuName = (wuIdentifier + "_eacirc_" + projectName + "_" + itostr(algorithmsRounds[k][0]) + "_" + algorithmName +
					+ "_r" + itostr(algorithmsRounds[k][l]) + "_" + itostr(numGenerations[i]) + "-gen");
				configName = wuName;
				configName.append(".xml");

				//Adding line into upload script 
				createWuMethod = createWuMethodPrototype;
				replaceInString(&createWuMethod , KEYWORD_METHOD_CREATE_WU , DEFAULT_METHOD_CREATE_WU_NAME);
				replaceInString(&createWuMethod , KEYWORD_WU_NAME , wuName);
				replaceInString(&createWuMethod , KEYWORD_CONFIG_PATH , (std::string)DIRECTORY_CFGS + configName);
				uploadScriptPosition = insertIntoScript(&uploadScriptSample , createWuMethodPrototype , createWuMethod , uploadScriptPosition);

				//Adding line into download script
				//Creation of directory for workunit results
				wuDirectory = DIRECTORY_RESULTS + projectName + "_" + algorithmName + "_r" + itostr(algorithmsRounds[k][l]) + "/";
				createDirectoryMethod = createDirectoryMethodPrototype;
				replaceInString(&createDirectoryMethod , KEYWORD_METHOD_CREATE_DIRECTORY , DEFAULT_METHOD_CREATE_DIRECTORY_NAME);
				replaceInString(&createDirectoryMethod , KEYWORD_DIRECTORY_PATH , wuDirectory);
				downloadScriptPosition = insertIntoScript(&downloadScriptSample , createDirectoryMethodPrototype , createDirectoryMethod , downloadScriptPosition);
				//Downloading workunit results
				downloadClonesMethod = downloadClonesMethodPrototype;
				replaceInString(&downloadClonesMethod , KEYWORD_METHOD_DOWNLOAD_CLONES , DEFAULT_METHOD_DOWNLOAD_CLONES_NAME);
				replaceInString(&downloadClonesMethod , KEYWORD_WU_NAME , wuName);
				replaceInString(&downloadClonesMethod , KEYWORD_WU_DIRECTORY , wuDirectory);
				downloadScriptPosition = insertIntoScript(&downloadScriptSample , downloadClonesMethodPrototype , downloadClonesMethod , downloadScriptPosition);

				//Set tags in config file - human readable description and number of generations
				notes.append(" function with " + itostr(algorithmsRounds[k][l]) + " rounds.");
				setXMLElementValue(root , PATH_EAC_NOTES , notes);
				setXMLElementValue(root , PATH_EAC_GENS , itostr(numGenerations[i]));
				n = eacNode->Clone();
				
				//Saving XML config file
				if(saveXMLFile(n , DIRECTORY_CFGS + configName) != STAT_OK)
					throw runtime_error("can't save file (directory must exist): " + (string)DIRECTORY_CFGS + configName);

				//Cleaning up (just to be sure)
				downloadClonesMethod.clear();
				wuDirectory.clear();
				createWuMethod.clear();
				createDirectoryMethod.clear();
				wuName.clear();
				notes.clear();
				configName.clear();
				algorithmName.clear();
				projectName.clear();
			}
		}
	}

	saveStringToFile(FILE_SCRIPT_UPLOAD , &uploadScriptSample);
	saveStringToFile(FILE_SCRIPT_DOWNLOAD , &downloadScriptSample);
}

std::string FileGenerator::readFileToString(std::string path) {
	std::ifstream file(path , std::ios::in);
	if(!file.is_open()) throw runtime_error("can't open input file: " + path);
	std::stringstream buffer;
	buffer << file.rdbuf();
	file.close();
	if(file.is_open()) throw runtime_error("can't close input file: " + path);
	return buffer.str();
}

void FileGenerator::saveStringToFile(std::string path , std::string * source) {
	std::ofstream file(path , std::ios::out);
	if(!file.is_open()) throw runtime_error("can't open output file: " + path);
	file << *source;
	file.close();
	if(file.is_open()) throw runtime_error("can't close output file: " + path);
	source->clear();
}

std::string FileGenerator::getMethodPrototype(std::string source , std::string methodName) {
	int pos = source.find(methodName);
	if(pos == -1) throw runtime_error("can't find method name: " + methodName);
	std::string methodPrototype = source.substr(pos , source.find(DEFAULT_SCRIPT_LINE_SEPARATOR , pos) - pos + 1);
	return methodPrototype;
}

void FileGenerator::replaceInString(std::string * target , std::string replace , std::string instead) {
	int pos = target->find(replace);
	if(pos == -1) throw runtime_error("can't find string to be replaced: " + replace);
	target->replace(pos , replace.length() , instead);
}

int FileGenerator::insertIntoScript(std::string * target , std::string methodPrototype , std::string toInsert , int position) {
	if(position == target->find(methodPrototype)) {
		target->replace(position , methodPrototype.length() , toInsert);
		//return (position + toInsert.length() + 1);
		return (position + toInsert.length() + 2);//test
	} else {
		toInsert.push_back('\n');
		//toInsert.insert(0 , "\t");
		toInsert.push_back('\t');//test
		target->insert(position , toInsert);
		return (position + toInsert.length());
	}
}

