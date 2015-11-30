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
    std::deque<Config> configs = parser->getConfigs();
    int project = parser->getProject();
    int clones = parser->getClones();
    int boincProjectID = parser->getBoincProjectID();
    std::string wuIdentifier = parser->getWuIdentifier();

    TiXmlNode * root = parser->getRoot();
    TiXmlNode * eacNode = NULL;
    eacNode = getXMLElement(root , PATH_EACIRC);

    //Preparing variables for script handling
    bool createWuFirstInsert = true;
    bool downloadRemDirFirstInsert = true;
    bool extractDeleteArchiveFirstInsert = true;

    std::string uploadScriptSample = Utils::readFileToString((std::string)DIRECTORY_SCRIPT_SAMPLES + (std::string)FILE_SCRIPT_UPLOAD_SAMPLE);
    oneclickLogger << FileLogger::LOG_INFO << "file " << FILE_SCRIPT_UPLOAD_SAMPLE << " was loaded into memory\n";
    std::string downloadScriptSample = Utils::readFileToString((std::string)DIRECTORY_SCRIPT_SAMPLES + (std::string)FILE_SCRIPT_DOWNLOAD_SAMPLE);
    oneclickLogger << FileLogger::LOG_INFO << "file " << FILE_SCRIPT_DOWNLOAD_SAMPLE << " was loaded into memory\n";

    //Fixing newlines
    Utils::fixNewlines(uploadScriptSample);
    Utils::fixNewlines(downloadScriptSample);

    replaceInString(uploadScriptSample, KEYWORD_CLONES, Utils::itostr(clones));
    replaceInString(uploadScriptSample, KEYWORD_PROJECT_ID, Utils::itostr(boincProjectID));
    replaceInString(downloadScriptSample, KEYWORD_PROJECT_ID, Utils::itostr(boincProjectID));

    std::string createWuMethodPrototype = getMethodPrototype(uploadScriptSample, KEYWORD_METHOD_CREATE_WU);
    std::string downloadRemDirMethodPrototype = getMethodPrototype(downloadScriptSample, KEYWORD_METHOD_DOWNLOAD_REM_DIR);
    std::string extractDeleteArchiveMethodPrototype = getMethodPrototype(downloadScriptSample, KEYWORD_METHOD_EXTRACT_DELETE_ARCHIVE);

    int uploadScriptPosition = uploadScriptSample.find(createWuMethodPrototype);
    int downloadScriptPosition = min(downloadScriptSample.find(downloadRemDirMethodPrototype),
        downloadScriptSample.find(extractDeleteArchiveMethodPrototype));

    //Variables declaration
    TiXmlNode * n = NULL;
    int invalidConfigs = 0;
    std::string configName , wuName , notes , projectName , wuID;
    std::string algorithmName , archiveName , createWuMethod;
    std::string downloadRemDirMethod , extractDeleteArchiveMethod;

    std::vector<std::pair<std::string , int>> configSettings;
    int algorithmConstant = 0;
    int algorithmRounds = 0;

    Utils::createDirectory(DIRECTORY_CFGS);

    oneclickLogger << FileLogger::LOG_INFO << "started generating config files\n";

    //Generating files from configs.
    for(unsigned i = 0 ; i < configs.size() ; i++) {
        configSettings = configs[i].getSettings();
        wuID = wuIdentifier;
        algorithmConstant = configs[i].getAlgorithm();
        algorithmRounds = configs[i].getAlgorithmRound();

        //Setting workunit name and notes 
        //In file, algorithm constant and rounds are set
        OneclickConstants::setAlgorithmSpecifics(root , project , algorithmConstant ,
            algorithmRounds , projectName , algorithmName);

        notes = projectName;
        notes.append(": " + algorithmName + " - " + Utils::itostr(algorithmRounds) + " rounds");
        wuName = (wuName + Utils::getDate() + "_" + OneclickConstants::getBoincProjectShort(boincProjectID) + "_" + projectName + 
            + "_a" + Utils::itostr(algorithmConstant , 3) + "r" + Utils::itostr(algorithmRounds , 2));

        //Adding settings to config
        for(unsigned k = 0 ; k < configSettings.size() ; k++) {
            if(setXMLElementValue(root , configSettings[k].first , Utils::itostr(configSettings[k].second)) == STAT_INVALID_ARGUMETS)
                throw std::runtime_error("invalid requested path in config: " + configSettings[k].first);
            if(wuID.length() > 0) wuID.append("_");
            wuID.append(Utils::itostr(configSettings[k].second));
        }
        if(wuID.length() > 0) wuName.append("_" + wuID);
        configName = wuName + ".xml";

        //Check for workunit name length
        if (wuName.length() <= BOINC_MAX_WU_NAME_LENGTH) {
            //Adding lines to upload and download script
            createWuMethod = createWuMethodPrototype;
            replaceInString(createWuMethod, KEYWORD_METHOD_CREATE_WU, DEFAULT_METHOD_CREATE_WU_NAME);
            replaceInString(createWuMethod, KEYWORD_WU_NAME, wuName);
            replaceInString(createWuMethod, KEYWORD_CONFIG_PATH, DIRECTORY_CFGS + configName);

            uploadScriptPosition = insertIntoScript(uploadScriptSample, createWuMethodPrototype,
                createWuMethod, uploadScriptPosition, createWuFirstInsert);
            createWuFirstInsert = false;

            downloadRemDirMethod = downloadRemDirMethodPrototype;
            replaceInString(downloadRemDirMethod, KEYWORD_METHOD_DOWNLOAD_REM_DIR, DEFAULT_METHOD_DOWNLOAD_REM_DIR_NAME);
            replaceInString(downloadRemDirMethod, KEYWORD_REM_DIR_NAME, wuName);

            downloadScriptPosition = insertIntoScript(downloadScriptSample, downloadRemDirMethodPrototype,
                downloadRemDirMethod, downloadScriptPosition, downloadRemDirFirstInsert);
            downloadRemDirFirstInsert = false;

            archiveName = wuName + ".zip";
            extractDeleteArchiveMethod = extractDeleteArchiveMethodPrototype;
            replaceInString(extractDeleteArchiveMethod, KEYWORD_METHOD_EXTRACT_DELETE_ARCHIVE, DEFAULT_METHOD_EXTRACT_DELETE_ARCHIVE_NAME);
            replaceInString(extractDeleteArchiveMethod, KEYWORD_ARCHIVE_NAME, archiveName);

            downloadScriptPosition = insertIntoScript(downloadScriptSample, extractDeleteArchiveMethodPrototype,
                extractDeleteArchiveMethod, downloadScriptPosition, extractDeleteArchiveFirstInsert);
            extractDeleteArchiveFirstInsert = false;

            if (wuID.length() > 0) {
                notes.append(" [");
                notes.append(wuID);
                notes.append("]");
            }

            //Setting notes in config
            if (setXMLElementValue(root, PATH_EAC_NOTES, notes) == STAT_INVALID_ARGUMETS)
                throw std::runtime_error("invalid requested path in config: " + (std::string)PATH_EAC_NOTES);
            n = eacNode->Clone();

            //Saving config to file
            if (saveXMLFile(n, DIRECTORY_CFGS + configName) != STAT_OK)
                throw std::runtime_error("can't save file: " + (std::string)DIRECTORY_CFGS + configName);

            n = NULL;
        }
        else {
            oneclickLogger << FileLogger::LOG_WARNING << "job " << wuName << " wasn't generated: name is too long\n";
            invalidConfigs++;
        }

        //Cleanup
        downloadRemDirMethod.clear() ; archiveName.clear() ; createWuMethod.clear() ; extractDeleteArchiveMethod.clear();
        wuName.clear() ; notes.clear() ; configName.clear() ; algorithmName.clear() ; projectName.clear(); wuID.clear();

    } // End of config generation

    if ((configs.size() - invalidConfigs) == 0) throw std::runtime_error("no valid configs could be generated");

    oneclickLogger << FileLogger::LOG_INFO << Utils::itostr(configs.size() - invalidConfigs) << " configs were generated\n";

    std::string uploadScriptName;
    std::string downloadScriptName;

    if(wuIdentifier.length() > 0) {
        uploadScriptName.append(wuIdentifier + "_");
        downloadScriptName.append(wuIdentifier + "_");
    }
    uploadScriptName.append(FILE_SCRIPT_UPLOAD);
    downloadScriptName.append(FILE_SCRIPT_DOWNLOAD);

    //Saving upload and download script to file
    Utils::saveStringToFile(uploadScriptName , uploadScriptSample);
    oneclickLogger << FileLogger::LOG_INFO << "created file " << uploadScriptName << "\n";
    uploadScriptSample.erase();
    Utils::saveStringToFile(downloadScriptName , downloadScriptSample);
    oneclickLogger << FileLogger::LOG_INFO << "created file " << downloadScriptName << "\n";
    downloadScriptSample.erase();
}

std::string FileGenerator::getMethodPrototype(const std::string & source , const std::string & methodName) {
    int pos = source.find(methodName);
    if(pos == -1) throw std::runtime_error("can't find method name: " + methodName);
    std::string methodPrototype = source.substr(pos , source.find(DEFAULT_SCRIPT_LINE_SEPARATOR , pos) - pos + 1);
    return methodPrototype;
}

void FileGenerator::replaceInString(std::string & target , const std::string & replace , const std::string & instead) {
    int pos = target.find(replace);
    if(pos == -1) throw std::runtime_error("can't find string to be replaced: " + replace);
    target.replace(pos , replace.length() , instead);
}

int FileGenerator::insertIntoScript(std::string & target , const std::string & methodPrototype , std::string & toInsert , int position , bool firstInsert) {
    std::stringstream tmp;
    tmp << std::endl;
    std::string endlChar = tmp.str();

    if(firstInsert) {
        target.replace(position , methodPrototype.length() , toInsert);
        return (position + toInsert.length() + 5); //5 = newline + 4 spaces
    } else {
        toInsert.push_back('\n');
		//Inserting tab... with spaces!
		toInsert.push_back(' '); toInsert.push_back(' '); toInsert.push_back(' '); toInsert.push_back(' ');
        target.insert(position , toInsert);
        return (position + toInsert.length());
    }
}