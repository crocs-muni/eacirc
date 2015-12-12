#include "XMLProcessor.h"
#include "generators/QuantumRndGen.h"
#include "generators/BiasRndGen.h"
#include <typeinfo>

void LoadConfigScript(TiXmlNode* pRoot, SETTINGS *pSettings) {
    // parsing EACIRC/NOTES
    pSettings->notes = getXMLElementValue(pRoot,"NOTES");

    // parsing EACIRC/MAIN
    pSettings->main.circuitType = atoi(getXMLElementValue(pRoot,"MAIN/CIRCUIT_REPRESENTATION").c_str());
    pSettings->main.projectType = atoi(getXMLElementValue(pRoot,"MAIN/PROJECT").c_str());
    pSettings->main.evaluatorType = atoi(getXMLElementValue(pRoot,"MAIN/EVALUATOR").c_str());
    pSettings->main.evaluatorPrecision = atoi(getXMLElementValue(pRoot,"MAIN/EVALUATOR_PRECISION").c_str());
    pSettings->main.recommenceComputation = (atoi(getXMLElementValue(pRoot,"MAIN/RECOMMENCE_COMPUTATION").c_str())) ? true : false;
    pSettings->main.loadInitialPopulation = (atoi(getXMLElementValue(pRoot,"MAIN/LOAD_INITIAL_POPULATION").c_str())) ? true : false;
    pSettings->main.numGenerations = atol(getXMLElementValue(pRoot,"MAIN/NUM_GENERATIONS").c_str());
    pSettings->main.saveStateFrequency = atol(getXMLElementValue(pRoot,"MAIN/SAVE_STATE_FREQ").c_str());
    pSettings->main.circuitSizeOutput = atoi(getXMLElementValue(pRoot,"MAIN/CIRCUIT_SIZE_OUTPUT").c_str());
    pSettings->main.circuitSizeInput = atoi(getXMLElementValue(pRoot,"MAIN/CIRCUIT_SIZE_INPUT").c_str());

    // parsing EACIRC/OUTPUTS
    pSettings->outputs.verbosity = atoi(getXMLElementValue(pRoot,"OUTPUTS/VERBOSITY").c_str());
    pSettings->outputs.intermediateCircuits = (atoi(getXMLElementValue(pRoot,"OUTPUTS/INTERMEDIATE_CIRCUITS").c_str())) ? true : false;
    pSettings->outputs.allowPrunning = (atoi(getXMLElementValue(pRoot,"OUTPUTS/ALLOW_PRUNNING").c_str())) ? true : false;
    pSettings->outputs.saveTestVectors = (atoi(getXMLElementValue(pRoot,"OUTPUTS/SAVE_TEST_VECTORS").c_str())) ? true : false;
    pSettings->outputs.fractionFile = (atoi(getXMLElementValue(pRoot,"OUTPUTS/FRACTION_FILE").c_str())) ? true : false;

    // parsing EACIRC/RANDOM
    pSettings->random.useFixedSeed = (atoi(getXMLElementValue(pRoot,"RANDOM/USE_FIXED_SEED").c_str())) ? true : false;
    istringstream(getXMLElementValue(pRoot,"RANDOM/SEED")) >> pSettings->random.seed;
    pSettings->random.biasRndGenFactor = atoi(getXMLElementValue(pRoot,"RANDOM/BIAS_RNDGEN_FACTOR").c_str());
    pSettings->random.useNetShare = atoi(getXMLElementValue(pRoot,"RANDOM/USE_NET_SHARE").c_str()) ? true : false;
    pSettings->random.qrngPath = getXMLElementValue(pRoot,"RANDOM/QRNG_PATH");
    pSettings->random.qrngFilesMaxIndex = atoi(getXMLElementValue(pRoot,"RANDOM/QRNG_MAX_INDEX").c_str());

    // parsing EACIRC/CUDA
    pSettings->cuda.enabled = (atoi(getXMLElementValue(pRoot,"CUDA/ENABLED").c_str())) ? true : false;
    pSettings->cuda.block_size = atoi(getXMLElementValue(pRoot,"CUDA/BLOCK_SIZE").c_str());

    // parsing EACIRC/GA
    pSettings->ga.evolutionOff = atoi(getXMLElementValue(pRoot,"GA/EVOLUTION_OFF").c_str()) ? true : false;
    pSettings->ga.popupationSize = atoi(getXMLElementValue(pRoot,"GA/POPULATION_SIZE").c_str());
    pSettings->ga.replacementSize = atoi(getXMLElementValue(pRoot,"GA/REPLACEMENT_SIZE").c_str());
    pSettings->ga.probCrossing = (float) atof(getXMLElementValue(pRoot,"GA/PROB_CROSSING").c_str());
    pSettings->ga.probMutation = (float) atof(getXMLElementValue(pRoot,"GA/PROB_MUTATION").c_str());
    pSettings->ga.mutateFunctions = atoi(getXMLElementValue(pRoot,"GA/MUTATE_FUNCTIONS").c_str()) ? true : false;
    pSettings->ga.mutateConnectors = atoi(getXMLElementValue(pRoot,"GA/MUTATE_CONNECTORS").c_str()) ? true : false;

    // parsing EACIRC/TEST_VECTORS
    pSettings->testVectors.inputLength = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/INPUT_LENGTH").c_str());
    pSettings->testVectors.outputLength = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/OUTPUT_LENGTH").c_str());
    pSettings->testVectors.setSize = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/SET_SIZE").c_str());
    pSettings->testVectors.setChangeFrequency = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/SET_CHANGE_FREQ").c_str());
    pSettings->testVectors.evaluateEveryStep = (atoi(getXMLElementValue(pRoot,"TEST_VECTORS/EVALUATE_EVERY_STEP").c_str())) ? true : false;
    pSettings->testVectors.evaluateBeforeTestVectorChange = (atoi(getXMLElementValue(pRoot,"TEST_VECTORS/EVALUATE_BEFORE_TEST_VECTOR_CHANGE").c_str())) ? true : false;
    // compute extra info
    if (pSettings->testVectors.setChangeFrequency == 0) {
        pSettings->testVectors.numTestSets = 1;
    } else {
        pSettings->testVectors.numTestSets = pSettings->main.numGenerations / pSettings->testVectors.setChangeFrequency;
    }
}

int saveXMLFile(TiXmlNode* pRoot, string filename) {
    TiXmlDocument doc;
    TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );
    doc.LinkEndChild(decl);
    doc.LinkEndChild(pRoot);
    bool result = doc.SaveFile(filename.c_str());
    if (!result) {
        mainLogger.out(LOGGER_ERROR) << "Cannot write XML file (" << filename << ").";
        return STAT_FILE_OPEN_FAIL;
    }
    return STAT_OK;
}

int loadXMLFile(TiXmlNode*& pRoot, string filename) {
    TiXmlDocument doc(filename.c_str());
    if (!doc.LoadFile()) {
        mainLogger.out(LOGGER_ERROR) << "Could not load file (" << filename << ")." << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    TiXmlHandle hDoc(&doc);
    TiXmlElement* pElem=hDoc.FirstChildElement().Element();
    if (!pElem) {
        mainLogger.out(LOGGER_ERROR) << "No root element in XML (" << filename << ")." << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    pRoot = pElem->Clone();

    return STAT_OK;
}

string getXMLElementValue(TiXmlNode*& pRoot, string path) {
    TiXmlNode* pNode = getXMLElement(pRoot,path);
    if (pNode == NULL) {
        mainLogger.out(LOGGER_WARNING) << "no value at " << path << " in given XML." << endl;
        return "";
    }
    if (path.find('@') == path.npos) {
        // getting text node
        const char* text = pNode->ToElement()->GetText();
        return text != NULL ? string(text) : "";
    } else {
        // getting attribute
        string attrName = path.substr(path.find('@')+1,path.length()-path.find('@')-1).c_str();
        const char* attrValue = pNode->ToElement()->Attribute(attrName.c_str());
        if (attrValue == NULL) {
            mainLogger.out(LOGGER_WARNING) << "there is no attribute named " << attrName << "." << endl;
            return "";
        }
        return string(attrValue);
    }
}

int setXMLElementValue(TiXmlNode*& pRoot, string path, const string& value) {
    TiXmlNode* pNode = getXMLElement(pRoot,path);
    if (pNode == NULL) {
        mainLogger.out(LOGGER_WARNING) << "no value at " << path << " in given XML." << endl;
        return STAT_INVALID_ARGUMETS;
    }
    if (path.find('@') == path.npos) {
        // setting text node
        TiXmlText* pText = pNode->FirstChild()->ToText();
        if (pText == NULL) {
            mainLogger.out(LOGGER_WARNING) << "node at " << path << " is not a leaf in XML." << endl;
            return STAT_INVALID_ARGUMETS;
        }
        pText->SetValue(value.c_str());
    } else {
        // setting attribute
        pNode->ToElement()->SetAttribute(path.substr(path.find('@')+1,path.length()-path.find('@')-1).c_str(),value.c_str());
    }
    return STAT_OK;
}

TiXmlNode* getXMLElement(TiXmlNode* pRoot, string path) {
    TiXmlHandle handle(pRoot);
    path = path.substr(0,path.rfind("@"));
    if (!path.empty() && path.find_last_of('/') == path.length()-1) {
        path.erase(--path.end());
    }
    if (!path.empty()) {
        istringstream xpath(path);
        string nodeValue;
        while (!xpath.eof()) {
            getline(xpath,nodeValue,'/');
            handle = handle.FirstChild(nodeValue.c_str());
        }
    }
    return handle.Node();
}
