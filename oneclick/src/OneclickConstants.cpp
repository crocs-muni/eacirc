#include "OneclickConstants.h"

FileLogger oneclickLogger(FILE_LOG);

void OneclickConstants::setAlgorithmSpecifics(TiXmlNode * root , int projectConstant , int algorithmConstant , int rounds , std::string & projectName , std::string & algorithmName) {
	switch(projectConstant) {
	case PROJECT_ESTREAM:
		//Setting of project name
		projectName.append("eStream");

		//Setting of algorithm name - project has to have implemeted interface with 
		//std::string projectToString(int function) method
		algorithmName.append(EstreamCiphers::estreamToString(algorithmConstant));

		//Check for existence of specified algorithm
		if(algorithmName.compare("(unknown stream cipher)") == 0) {
			throw std::runtime_error("unknown algorithm constant: " + Utils::itostr(algorithmConstant));
		}

		//Setting values into new config.xml file, paths to corresponding tags
		//should be defined as constants.
		if(setXMLElementValue(root , PATH_ESTR_ALG , Utils::itostr(algorithmConstant)) == STAT_INVALID_ARGUMETS)
			throw std::runtime_error("invalid requested path in config: " + (std::string)PATH_ESTR_ALG);
		if(setXMLElementValue(root , PATH_ESTR_RND , Utils::itostr(rounds)) == STAT_INVALID_ARGUMETS)
			throw std::runtime_error("invalid requested path in config: " + (std::string)PATH_ESTR_RND);
		break;

	case PROJECT_SHA3:
		projectName.append("SHA3");
		algorithmName.append(Sha3Functions::sha3ToString(algorithmConstant));
		if(algorithmName.compare("(unknown hash function)") == 0) {
			throw std::runtime_error("unknown algorithm constant: " + Utils::itostr(algorithmConstant));
		}
		if(setXMLElementValue(root , PATH_SHA3_ALG , Utils::itostr(algorithmConstant)) == STAT_INVALID_ARGUMETS)
			throw std::runtime_error("invalid requested path in config: " + (std::string)PATH_SHA3_ALG);
		if(setXMLElementValue(root , PATH_SHA3_RND , Utils::itostr(rounds)) == STAT_INVALID_ARGUMETS)
			throw std::runtime_error("invalid requested path in config: " + (std::string)PATH_SHA3_RND);
		break;
	default:
		throw std::runtime_error("invalid project constant set at: " + (std::string)PATH_EAC_PROJECT);
	}
}
