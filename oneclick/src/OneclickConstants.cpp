#include "OneclickConstants.h"

FileLogger oneclickLogger(FILE_LOG);

void OneclickConstants::setAlgorithmSpecifics(TiXmlNode * root , int projectConstant , int algorithmConstant , int rounds , std::string & projectName , std::string & algorithmName) {
    switch(projectConstant) {
    case PROJECT_ESTREAM:
        //Setting of project name
        projectName.append(EACIRC_PROJECT_NAME_ESTREAM);

        //Setting of algorithm name - project has to have implemeted interface with 
        //std::string projectToString(int function) method
        algorithmName.append(EstreamCiphers::estreamToString(algorithmConstant));

        //Check for existence of specified algorithm
        if(algorithmName.compare("(unknown stream cipher)") == 0) {
            throw std::runtime_error("unknown algorithm constant: " + Utils::itostr(algorithmConstant));
        }
        break;

    case PROJECT_SHA3:
        projectName.append(EACIRC_PROJECT_NAME_SHA3);
        algorithmName.append(Sha3Functions::sha3ToString(algorithmConstant));
        if(algorithmName.compare("(unknown hash function)") == 0) {
            throw std::runtime_error("unknown algorithm constant: " + Utils::itostr(algorithmConstant));
        }
        break;

    case PROJECT_CAESAR:
        projectName.append(EACIRC_PROJECT_NAME_CAESAR);
        algorithmName.append(CaesarCiphers::caesarToString(algorithmConstant));
        if (algorithmName.compare("(unknown cipher)") == 0) {
            throw std::runtime_error("unknown algorithm constant: " + Utils::itostr(algorithmConstant));
        }
        break;

    case PROJECT_FILE_DISTINGUISHER:
        projectName.append(EACIRC_PROJECT_NAME_FILES);
        break;

    default:
        throw std::runtime_error("invalid project constant set at: " + (std::string)PATH_EAC_PROJECT);
    }

    //Setting algoritm constant and rounds into config file on project specific paths
    //Every project should have case in methods getProjectAlgorithmPath and getProjectRoundPath
    if (setXMLElementValue(root, getProjectAlgorithmPath(projectConstant), Utils::itostr(algorithmConstant)) == STAT_INVALID_ARGUMETS)
        throw std::runtime_error("invalid requested path in config: " + getProjectAlgorithmPath(projectConstant));
    if (setXMLElementValue(root, getProjectRoundPath(projectConstant), Utils::itostr(rounds)) == STAT_INVALID_ARGUMETS)
        throw std::runtime_error("invalid requested path in config: " + getProjectRoundPath(projectConstant));
}

std::string OneclickConstants::getProjectAlgorithmPath(int projectConstant) {
    switch (projectConstant) {
    case PROJECT_ESTREAM:   return PATH_ESTR_ALG;
    case PROJECT_SHA3:      return PATH_SHA3_ALG;
    case PROJECT_CAESAR:    return PATH_CAESAR_ALG;
    //Add new project case below. All values should be defined as constants.
    default:
        throw std::runtime_error("invalid project constant set at: " + (std::string)PATH_EAC_PROJECT);
    }
}

std::string OneclickConstants::getProjectRoundPath(int projectConstant) {
    switch (projectConstant) {
    case PROJECT_ESTREAM:   return PATH_ESTR_RND;
    case PROJECT_SHA3:      return PATH_SHA3_RND;
    case PROJECT_CAESAR:    return PATH_CAESAR_RND;
    //Add new project case below. All values should be defined as constants.
    default:
        throw std::runtime_error("invalid project constant set at: " + (std::string)PATH_EAC_PROJECT);
    }
}

int OneclickConstants::getBoincProjectID(const std::string & logicalProjectName) {
    if (logicalProjectName.compare(BOINC_PROJECT_NAME_EACIRC_MAIN) == 0) return BOINC_PROJECT_ID_EACIRC_MAIN;
    if (logicalProjectName.compare(BOINC_PROJECT_NAME_EACIRC_DEV) == 0) return BOINC_PROJECT_ID_EACIRC_DEV;
    if (logicalProjectName.compare(BOINC_PROJECT_NAME_EACIRC_CUDA) == 0) return BOINC_PROJECT_ID_EACIRC_CUDA;

    //If method reaches this point, no known project was set, throwing error!
    throw std::runtime_error("invalid/empty BOINC project set at: " + (std::string)PATH_OC_BOINC_PROJECT);
}

std::string OneclickConstants::getBoincProjectShort(int projectID) {
    switch (projectID) {
    case BOINC_PROJECT_ID_EACIRC_MAIN:
        return BOINC_PROJECT_SHORT_EACIRC_MAIN;
        break;
    case BOINC_PROJECT_ID_EACIRC_DEV:
        return BOINC_PROJECT_SHORT_EACIRC_DEV;
        break;
    case BOINC_PROJECT_ID_EACIRC_CUDA:
        return BOINC_PROJECT_SHORT_EACIRC_CUDA;
        break;
    default: //This shouldn't happen, argument is always value returned from getBoincProjectID
        throw std::runtime_error("unknown project ID: " + Utils::itostr(projectID));
    }
}
