#include "ConfigParser.h"

ConfigParser::ConfigParser(std::string path) {
    if(loadXMLFile(root , path) == STAT_FILE_OPEN_FAIL) throw std::runtime_error("can't open XML file: " + path);
    oneclickLogger << FileLogger::LOG_INFO << "started parsing config file\n";

    wuIdentifier = getXMLElementValue(root , PATH_OC_WU_ID);
    checkWUIdentifier();

    //Get BOINC project constant => logical string from tag have to be converted to BOINC project constant
    //in method getBoincProjectID
    boincProjectID = OneclickConstants::getBoincProjectID(getXMLElementValue(root, PATH_OC_BOINC_PROJECT));

    clones = getXMLValue(PATH_OC_CLONES);
    project = getXMLValue(PATH_EAC_PROJECT);
    setConfigs();

    oneclickLogger << FileLogger::LOG_INFO << "finished parsing config file\n";
}

ConfigParser::~ConfigParser() {
    delete root;
}

void ConfigParser::setConfigs() {
    algorithm_rounds_v algorithmsRounds = createAlgorithmsRounds();

    //Creating initial configs with nothing but algorithm and rounds set
    for(unsigned i = 0 ; i < algorithmsRounds.size() ; i++) {
        for(unsigned k = 0 ; k < algorithmsRounds[i].second.size() ; k++) {
            Config cfg(algorithmsRounds[i].first , algorithmsRounds[i].second[k]);
            configs.push_back(cfg);
        }
    }

    //Getting additional settings from Oneclick Config
    std::vector<std::pair<std::string , std::vector<int>>> settings = parseChildrenTags(PATH_OC_ADD_SETT , PATH_OC_ATT_ADD_SETT);
    unsigned originalSize = 0;

    //Creating every possible combination of settings algorithms and rounds
    for(unsigned current = 0 ; current < settings.size() ; current++) {
        originalSize = configs.size();

        for(unsigned i = 0 ; i < originalSize ; i++) {
            Config originalConfig(configs.front());
            configs.pop_front();

            for(unsigned k = 0 ; k < settings[current].second.size() ; k++) {
                Config newConfig(originalConfig);
                newConfig.addSetting(settings[current].first , settings[current].second[k]);
                configs.push_back(newConfig);
            }
        }
    }
}

ConfigParser::algorithm_rounds_v ConfigParser::createAlgorithmsRounds() {
    algorithm_rounds_v algorithmsRounds;
    std::vector<int> rounds = getMultipleXMLValues(PATH_OC_RNDS);
    std::vector<int> algorithms = getMultipleXMLValues(PATH_OC_ALGS);
    attribute_values_v tempSpecRounds = parseChildrenTags(PATH_OC_SPEC_RNDS , PATH_OC_ATT_SPEC_RNDS);
    algorithm_rounds_v specificRounds;

    //Conversion of tempSpecRounds into desired format
    algorithm_rounds singleAlg;
    for(unsigned i = 0 ; i < tempSpecRounds.size() ; i++) {
        try {
            singleAlg.first = std::stoi(tempSpecRounds[i].first);
            singleAlg.second = tempSpecRounds[i].second;
            specificRounds.push_back(singleAlg);
            singleAlg.second.clear();
        } catch(std::invalid_argument) { throw std::runtime_error("attribute algorithm in SPECIFIC_ROUNDS contains invalid characters");
        } catch(std::out_of_range) {     throw std::runtime_error("attribute algorithm constant in SPECIFIC_ROUNDS is out of range"); }
    }

    //Checking if ALGORITHMS and ROUNDS options were set
    //Needed only when specific rounds weren't set
    //If not, use values from main EACIRC config
    //If correct values are not set in EAC config this will fail!
    if (algorithms.size() == 0 && specificRounds.size() == 0)
        algorithms.push_back(getXMLValue(OneclickConstants::getProjectAlgorithmPath(project)));
    if (rounds.size() == 0 && specificRounds.size() == 0)
        rounds.push_back(getXMLValue(OneclickConstants::getProjectRoundPath(project)));


    //Saving algorithms into algorithm_rounds_v structure
    for(unsigned i = 0 ; i < algorithms.size() ; i++) {
        singleAlg.first = algorithms[i];
        for(unsigned k = 0 ; k < rounds.size() ; k++) {
            singleAlg.second.push_back(rounds[k]);
        }
        algorithmsRounds.push_back(singleAlg);
        singleAlg.second.clear();
    }
    //Adding algorithms with different rounds set
    for(unsigned i = 0 ; i < specificRounds.size() ; i++) {
        algorithmsRounds.push_back(specificRounds[i]);
    }
    sort2D(algorithmsRounds);
    return algorithmsRounds;
}

ConfigParser::attribute_values_v ConfigParser::parseChildrenTags(const std::string & parentPath , const std::string & childAttribute) {
    attribute_values_v parsedTags;
    attribute_values singleTag;
    std::vector<int> valuesInTag;
    const char * attributeValue;
    const char * tagText;

    TiXmlNode * parentNode = getXMLElement(root , parentPath);
    TiXmlElement * childElement;
    if(parentNode == NULL) return parsedTags;

    if(parentNode->FirstChild()) {
        childElement = parentNode->FirstChildElement();
        for(;;) {
            attributeValue = childElement->Attribute(childAttribute.c_str());
            if(attributeValue == NULL) throw std::runtime_error("child of tag " + parentPath + " doesn't have attribute \"" + childAttribute + "\"");
            tagText = childElement->GetText();
            if(strlen(attributeValue) > 0 && tagText != NULL) {
                valuesInTag = parseStringValue(tagText , parentPath);
                sort(valuesInTag);
                singleTag.first = attributeValue;
                singleTag.second = valuesInTag;
                parsedTags.push_back(singleTag);
            }
            if(!childElement->NextSiblingElement()) break;
            childElement = childElement->NextSiblingElement();
        }
    }
    return parsedTags;
}

int ConfigParser::getXMLValue(const std::string & path) {
    std::string temp = getXMLElementValue(root , path);
    int result = 0;

    if(temp.length() == 0) throw std::runtime_error("empty or nonexistent XML tag: " + path);

    try {
        result = std::stoi(temp , nullptr);
        return result;
    } catch(std::invalid_argument e) {
        throw std::runtime_error("invalid value in xml element: " + path);
    } catch(std::out_of_range e) {
        throw std::runtime_error("value out of range in xml element: " + path);
    }
}

std::vector<int> ConfigParser::getMultipleXMLValues(const std::string & path) {
    std::string elementValue = getXMLElementValue(root , path);
    std::string temp;
    std::vector<int> values;
    values = parseStringValue(elementValue , path);
    sort(values);
    return values;
}

std::vector<int> ConfigParser::parseStringValue(const std::string & elementValue , const std::string & path) {
    std::string temp;
    std::vector<int> result;

    for(unsigned i = 0; i < elementValue.length(); i++) {
        if((elementValue[i] < 48 || elementValue[i] > 57) && elementValue[i] != ' ' && elementValue[i] != '-') {
            throw std::runtime_error("invalid characters in xml element: " + path);
        }
    }

    for(unsigned i = 0; i < elementValue.length(); i++) {
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
            i = parseRange(temp , elementValue , i , result , path);
            break;
        default:
            temp.push_back(elementValue[i]);
            break;
        }
    }

    return result;
}

int ConfigParser::parseRange(std::string & temp , const std::string & elementValue , unsigned iterator , std::vector<int> & result , const std::string & path) {
    if(temp.length() == 0) { throw std::runtime_error("invalid structure of xml element: " + path); }
    int bottom = atoi(temp.c_str());
    int top = 0;
    temp.clear();
    iterator++;
    for( ; iterator < elementValue.length(); iterator++) {
        if(iterator == (elementValue.length() - 1)) {
            temp.push_back(elementValue[iterator]);
            top = atoi(temp.c_str());
            temp.clear();
            break;
        }
        if(elementValue[iterator] != ' ') { temp.push_back(elementValue[iterator]); } else {
            if(temp.length() == 0) { throw std::runtime_error("invalid structure of xml element: " + path); }
            top = atoi(temp.c_str());
            temp.clear();
            break;
        }
    }
    for( ; bottom <= top; bottom++) { result.push_back(bottom); }
    return iterator;
}

void ConfigParser::sort(std::vector<int> & a , unsigned begin) {
    for(unsigned i = begin ; i < a.size() ; i++) {
        for(unsigned k = i ; k > begin ; k--) {
            if(a[k] < a[k - 1]) {
                std::iter_swap(a.begin() + k, a.begin() + k - 1);
            } else {
                break;
            }
        }
    }
    for(unsigned i = begin ; ; i++) {
        if(i >= a.size() - 1 || a.size() == 0) break;
        if(a[i] == a[i + 1]) {
            a.erase(a.begin() + i);
            i--;
        }
    }
}

void ConfigParser::sort2D(ConfigParser::algorithm_rounds_v & a) {
    //std::pair<int , std::vector<int>> temp;
    //sort
    for(unsigned i = 0 ; i < a.size() ; i++) {
        for(unsigned k = i ; k > 0 ; k--) {
            if(a[k].first < a[k - 1].first) {
                std::iter_swap(a.begin() + k, a.begin() + k - 1);
            } else {
                break;
            }
        }
    }
    //eliminate duplicities (the latter one survive)
    for(unsigned i = 0 ; ; i++) {
        if(i >= a.size() - 1 || a.size() == 0) break;
        if(a[i].first == a[i + 1].first) {
            a.erase(a.begin() + i);
            i--;
        }
    }
}

void ConfigParser::checkWUIdentifier() {
    if (wuIdentifier.length() > BOINC_MAX_WU_NAME_LENGTH)
        throw std::runtime_error("workunit identifier is too long");
    std::regex valid("^[A-Za-z0-9\\[\\]\\-_()]*$");
    if (!std::regex_match(wuIdentifier, valid))
        throw std::runtime_error("workunit identifier contains illegal characters");
}
