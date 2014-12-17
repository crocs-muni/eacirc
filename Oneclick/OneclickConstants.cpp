#include "OneclickConstants.h"


//When new project is added modify this method!
void setAlgorithmSpecifics(TiXmlNode * root , int projectConstant , int algorithmConstant , std::string * projectName , std::string * algorithmName) {
	switch(projectConstant) {
	case PROJECT_ESTREAM:
		//Setting of project name
		projectName->append("eStream");
		//Setting of algorithm name - project has to have implemeted interface
		//with std::string projectToString(int function) method
		algorithmName->append(estreamToString(algorithmConstant));
		//Check for existence of specified algorithm
		if(algorithmName->compare("(unknown stream cipher)") == 0) {
			throw runtime_error("unknown algorithm constant: " + itostr(algorithmConstant));
		}
		//Setting values into new config.xml file, paths to corresponding tags
		//should be defined as constants.
		setXMLElementValue(root , PATH_ESTR_ALG , itostr(algorithmConstant));
		setXMLElementValue(root , PATH_SHA3_ALG , itostr(algorithmConstant));
		break;
	case PROJECT_SHA3:
		projectName->append("SHA3");
		algorithmName->append(Sha3Interface::sha3ToString(algorithmConstant));
		if(algorithmName->compare("(unknown hash function)") == 0) {
			throw runtime_error("unknown algorithm constant: " + itostr(algorithmConstant));
		}
		setXMLElementValue(root , PATH_ESTR_ALG , itostr(algorithmConstant));
		setXMLElementValue(root , PATH_SHA3_ALG , itostr(algorithmConstant));
		break;
	default:
		throw runtime_error("invalid project constant set at: " + (string)PATH_EAC_PROJECT);
	}
}

std::string itostr(int x) {
	std::stringstream ss;
	ss << x;
	return ss.str();
}