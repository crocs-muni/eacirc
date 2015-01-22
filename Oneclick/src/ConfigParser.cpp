#include "ConfigParser.h"

ConfigParser::ConfigParser(std::string path) {
	if(loadXMLFile(root , path) == STAT_FILE_OPEN_FAIL) throw std::runtime_error("can't open XML file: " + path);
	oneclickLogger << FileLogger::LOG_INFO << "started parsing config file\n";
	wuIdentifier = getXMLElementValue(root , PATH_OC_WU_ID);
	clones = getXMLValue(PATH_OC_CLONES);
	numGenerations = getMultipleXMLValues(PATH_OC_NUM_GENS);
	project = getXMLValue(PATH_EAC_PROJECT);
	std::vector<int> rounds = getMultipleXMLValues(PATH_OC_RNDS);
	std::vector<int> algorithms = getMultipleXMLValues(PATH_OC_ALGS);
	std::vector<std::vector<int>> specificRounds = getSpecificRounds();
	setAlgorithmsRounds(rounds , algorithms , specificRounds);
	oneclickLogger << FileLogger::LOG_INFO << "finished parsing config file\n";
}

ConfigParser::~ConfigParser() {
	delete root;
}

int ConfigParser::getXMLValue(std::string path) {
	std::string temp = getXMLElementValue(root , path);
	if(temp.length() == 0) throw std::runtime_error("empty or nonexistent XML tag: " + path);

	for(int i = 0; i < temp.length(); i++) {
		if(temp[i] < 48 || temp[i] > 57) {
			throw std::runtime_error("invalid characters in xml element: " + path);
		}
	}
	return atoi(temp.c_str());
}

std::vector<int> ConfigParser::getMultipleXMLValues(std::string path) {
	std::string elementValue = getXMLElementValue(root , path);
	std::string temp;
	std::vector<int> values;
	values = parseStringValue(elementValue , path);
	sort(&values , 0);
	return values;
}

std::vector<std::vector<int>> ConfigParser::getSpecificRounds() {
	std::string path = PATH_OC_SPEC_RNDS;
	std::vector<std::vector<int>> values;
	std::vector<int> single;
	TiXmlNode * specRndsNode = getXMLElement(root , path);
	TiXmlElement * rndsElement;
	if(specRndsNode == NULL) return values;

	if(specRndsNode->FirstChild()) {
		rndsElement = specRndsNode->FirstChildElement();
		for(;;) {
			const char * alg = rndsElement->Attribute("algorithm");
			if(alg == NULL) throw std::runtime_error("tag ROUNDS don't have attribute \"algorithm\"");
			const char * rnds = rndsElement->GetText();
			if(strlen(alg) > 0 && rnds != NULL) {
				single = parseStringValue(rnds , path);
				single.insert(single.begin() , atoi(alg));
				sort(&single , 1);
				values.push_back(single);
				single.clear();
			}
			if(!rndsElement->NextSiblingElement()) break;
			rndsElement = rndsElement->NextSiblingElement();
		}
	}
	return sort2D(values);
}

int ConfigParser::parseRange(std::string * temp , std::string elementValue , int iterator , std::vector<int> * result , std::string path) {
	if(temp->length() == 0) { throw std::runtime_error("invalid structure of xml element: " + path); }
	int bottom = atoi(temp->c_str());
	int top = 0;
	temp->clear();
	iterator++;
	for(iterator; iterator < elementValue.length(); iterator++) {
		if(iterator == (elementValue.length() - 1)) {
			temp->push_back(elementValue[iterator]);
			top = atoi(temp->c_str());
			temp->clear();
			break;
		}
		if(elementValue[iterator] != ' ') { temp->push_back(elementValue[iterator]); } else {
			if(temp->length() == 0) { throw std::runtime_error("invalid structure of xml element: " + path); }
			top = atoi(temp->c_str());
			temp->clear();
			break;
		}
	}
	for(bottom; bottom <= top; bottom++) { result->push_back(bottom); }
	return iterator;
}

void ConfigParser::setAlgorithmsRounds(std::vector<int> rounds , std::vector<int> algorithms , std::vector<std::vector<int>> specificRounds) {
	std::vector<int> singleAlg;
	for(int i = 0 ; i < algorithms.size() ; i++) {
		singleAlg.push_back(algorithms[i]);
		for(int k = 0 ; k < rounds.size() ; k++) {
			singleAlg.push_back(rounds[k]);
		}
		algorithmsRounds.push_back(singleAlg);
		singleAlg.clear();
	}
	for(int i = 0 ; i < specificRounds.size() ; i++) {
		algorithmsRounds.push_back(specificRounds[i]);
	}
	algorithmsRounds = sort2D(algorithmsRounds);
}

std::vector<int> ConfigParser::parseStringValue(std::string elementValue , std::string path) {
	std::string temp;
	std::vector<int> result;

	for(int i = 0; i < elementValue.length(); i++) {
		if((elementValue[i] < 48 || elementValue[i] > 57) && elementValue[i] != ' ' && elementValue[i] != '-') {
			throw std::runtime_error("invalid characters in xml element: " + path);
		}
	}

	for(int i = 0; i < elementValue.length(); i++) {
		if(i == (elementValue.length() - 1)) {
			temp.push_back(elementValue[i]);
			result.push_back(atoi(temp.c_str()));
			temp.clear();
			break;
		}
		switch(elementValue[i]) {
		case ' ':
			result.push_back(atoi(temp.c_str()));
			temp.clear();
			break;
		case '-':
			i = parseRange(&temp , elementValue , i , &result , path);
			break;
		default:
			temp.push_back(elementValue[i]);
			break;
		}
	}

	return result;
}

void ConfigParser::sort(std::vector<int> * a , int begin) {
	for(int i = begin ; i < a->size() ; i++) {
		for(int k = i ; k > begin ; k--) {
			if(a->at(k) < a->at(k - 1)) {
				int temp = a->at(k);
				a->at(k) = a->at(k - 1);
				a->at(k - 1) = temp;
			} else {
				break;
			}
		}
	}
	for(int i = begin ; ; i++) {
		if(i >= a->size() - 1 || a->size() == 0) break;
		if(a->at(i) == a->at(i + 1)) {
			a->erase(a->begin() + i);
			i--;
		}
	}
}

std::vector<std::vector<int>> ConfigParser::sort2D(std::vector<std::vector<int>> a) {
	for(int i = 0 ; i < a.size() ; i++) {
		for(int k = i ; k > 0 ; k--) {
			if(a[k][0] < a[k - 1][0]) {
				std::vector<int> temp = a[k];
				a[k] = a[k - 1];
				a[k - 1] = temp;
			} else {
				break;
			}
		}
	}
	for(int i = 0 ; ; i++) {
		if(i >= a.size() - 1 || a.size() == 0) break;
		if(a[i][0] == a[i + 1][0]) {
			a.erase(a.begin() + i);
			i--;
		}
	}
	return a;
}

