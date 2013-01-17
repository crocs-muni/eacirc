#include "XMLProcessor.h"
#include "random_generator/QuantumRndGen.h"
#include "random_generator/BiasRndGen.h"
#include <typeinfo>

void LoadConfigScript(TiXmlNode *pRoot, BASIC_INIT_DATA* pBasicSettings) {
    //
    // PROGRAM VERSION AND DATE
    //
    pBasicSettings->computationDate = getXMLElementValue(pRoot,"INFO/DATE");
    pBasicSettings->swVersion = getXMLElementValue(pRoot,"INFO/VERSION");

    //
    // MAIN SETTINGS
    //
    pBasicSettings->projectType = atoi(getXMLElementValue(pRoot,"MAIN/PROJECT").c_str());
    pBasicSettings->recommenceComputation = (atoi(getXMLElementValue(pRoot,"MAIN/RECOMMENCE_COMPUTATION").c_str())) ? true : false;
    pBasicSettings->loadInitialPopulation = (atoi(getXMLElementValue(pRoot,"MAIN/LOAD_INITIAL_POPULATION").c_str())) ? true : false;
    pBasicSettings->gaConfig.numGenerations = atol(getXMLElementValue(pRoot,"MAIN/NUM_GENERATIONS").c_str());
    pBasicSettings->gaCircuitConfig.changeGalibSeedFrequency = atol(getXMLElementValue(pRoot,"MAIN/SAVE_STATE_FREQ").c_str());
    pBasicSettings->rndGen.QRBGSPath = getXMLElementValue(pRoot,"MAIN/QRNG_PATH");

    //
    // RANDOM GENERATOR SETTINGS
    //
    pBasicSettings->rndGen.useFixedSeed = (atoi(getXMLElementValue(pRoot,"RANDOM/USE_FIXED_SEED").c_str())) ? true : false;
    istringstream(getXMLElementValue(pRoot,"RANDOM/SEED")) >> pBasicSettings->rndGen.seed;
    pBasicSettings->rndGen.primaryRandomType = atoi(getXMLElementValue(pRoot,"RANDOM/PRIMARY_RANDOM_TYPE").c_str());
    pBasicSettings->rndGen.biasRndGenFactor = atoi(getXMLElementValue(pRoot,"RANDOM/BIAS_RNDGEN_FACTOR").c_str());

    //
    // GA SETTINGS
    //
    pBasicSettings->gaConfig.evolutionOff = atoi(getXMLElementValue(pRoot,"GA/EVOLUTION_OFF").c_str()) ? true : false;
    pBasicSettings->gaConfig.probMutationt = atof(getXMLElementValue(pRoot,"GA/PROB_MUTATION").c_str());
    pBasicSettings->gaConfig.probCrossing = atof(getXMLElementValue(pRoot,"GA/PROB_CROSSING").c_str());
    pBasicSettings->gaConfig.popupationSize = atoi(getXMLElementValue(pRoot,"GA/POPULATION_SIZE").c_str());

    //
    // GA CIRCUIT SETTINGS
    //
    pBasicSettings->gaCircuitConfig.numLayers = atoi(getXMLElementValue(pRoot,"CIRCUIT/NUM_LAYERS").c_str());
    pBasicSettings->gaCircuitConfig.numSelectorLayers = atoi(getXMLElementValue(pRoot,"CIRCUIT/NUM_SELECTOR_LAYERS").c_str());
    pBasicSettings->gaCircuitConfig.sizeLayer = atoi(getXMLElementValue(pRoot,"CIRCUIT/SIZE_LAYER").c_str());
    pBasicSettings->gaCircuitConfig.sizeOutputLayer = atoi(getXMLElementValue(pRoot,"CIRCUIT/SIZE_OUTPUT_LAYER").c_str());
    pBasicSettings->gaCircuitConfig.sizeInputLayer = atoi(getXMLElementValue(pRoot,"CIRCUIT/SIZE_INPUT_LAYER").c_str());
    pBasicSettings->gaCircuitConfig.numConnectors = atoi(getXMLElementValue(pRoot,"CIRCUIT/NUM_CONNECTORS").c_str());
    pBasicSettings->gaCircuitConfig.predictionMethod = atoi(getXMLElementValue(pRoot,"CIRCUIT/PREDICTION_METHOD").c_str());
    pBasicSettings->gaCircuitConfig.genomeSize = atoi(getXMLElementValue(pRoot,"CIRCUIT/GENOME_SIZE").c_str());
    pBasicSettings->gaCircuitConfig.allowPrunning = (atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOW_PRUNNING").c_str())) ? true : false;

    //
    // ALLOWED FUNCTIONS
    //
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_NOP] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_NOP").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_OR] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_OR").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_AND] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_AND").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_CONST] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_CONST").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_XOR] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_XOR").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_NOR] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_NOR").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_NAND] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_NAND").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_ROTL] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_ROTL").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_ROTR] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_ROTR").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_SUM] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_SUM").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_SUBS] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_SUBS").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_ADD] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_ADD").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_MULT] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_MULT").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_DIV] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_DIV").c_str());
    pBasicSettings->gaCircuitConfig.allowedFunctions[FNC_READX] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_READX").c_str());

    //
    // TEST VECTORS
    //
    pBasicSettings->gaCircuitConfig.testVectorLength = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/TEST_VECTOR_LENGTH").c_str());
    pBasicSettings->gaCircuitConfig.numTestVectors = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/NUM_TEST_VECTORS").c_str());
    pBasicSettings->gaCircuitConfig.testVectorChangeFreq = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/TEST_VECTOR_CHANGE_FREQ").c_str());
    pBasicSettings->gaCircuitConfig.testVectorChangeProgressive = (atoi(getXMLElementValue(pRoot,"TEST_VECTORS/TEST_VECTOR_CHANGE_PROGRESSIVE").c_str())) ? true : false;
    pBasicSettings->gaCircuitConfig.saveTestVectors = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/SAVE_TEST_VECTORS").c_str());
    pBasicSettings->gaCircuitConfig.evaluateEveryStep = (atoi(getXMLElementValue(pRoot,"TEST_VECTORS/EVALUATE_EVERY_STEP").c_str())) ? true : false;
    pBasicSettings->gaCircuitConfig.evaluateBeforeTestVectorChange = (atoi(getXMLElementValue(pRoot,"TEST_VECTORS/EVALUATE_BEFORE_TEST_VECTOR_CHANGE").c_str())) ? true : false;
}

int saveXMLFile(TiXmlNode* pRoot, string filename) {
    TiXmlDocument doc;
    TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );
    doc.LinkEndChild(decl);
    doc.LinkEndChild(pRoot);
    bool result = doc.SaveFile(filename.c_str());
    if (!result) {
        mainLogger.out() << "error: Cannot write XML file " << filename << ".";
        return STAT_FILE_OPEN_FAIL;
    }
    return STAT_OK;
}

int loadXMLFile(TiXmlNode*& pRoot, string filename) {
    TiXmlDocument doc(filename.c_str());
    if (!doc.LoadFile()) {
        mainLogger.out() << "error: Could not load file '" << filename << "'." << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    TiXmlHandle hDoc(&doc);
    TiXmlElement* pElem=hDoc.FirstChildElement().Element();
    if (!pElem) {
        mainLogger.out() << "error: No root element in XML (" << filename << ")." << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    pRoot = pElem->Clone();

    return STAT_OK;
}

string getXMLElementValue(TiXmlNode*& pRoot, string path) {
    TiXmlNode* pNode = getXMLElement(pRoot,path);
    if (pNode == NULL) {
        mainLogger.out() << "error: no value at " << path << " in given XML." << endl;
        return "";
    }
    if (path.find('@') == path.npos) {
        // getting text node
        return pNode->ToElement()->GetText();
    } else {
        // getting attribute
        string attrName = path.substr(path.find('@')+1,path.length()-path.find('@')-1).c_str();
        const char* attrValue = pNode->ToElement()->Attribute(attrName.c_str());
        if (attrValue == NULL) {
            mainLogger.out() << "error: there is no attribute named " << attrName << "." << endl;
            return "";
        }
        return string(attrValue);
    }
    return STAT_OK;
}

int setXMLElementValue(TiXmlNode*& pRoot, string path, const string& value) {
    TiXmlNode* pNode = getXMLElement(pRoot,path);
    if (pNode == NULL) {
        mainLogger.out() << "error: no value at " << path << " in given XML." << endl;
        return STAT_INCOMPATIBLE_PARAMETER;
    }
    if (path.find('@') == path.npos) {
        // setting text node
        TiXmlText* pText = pNode->FirstChild()->ToText();
        if (pText == NULL) {
            mainLogger.out() << "error: node at " << path << " is not a leaf in XML." << endl;
            return STAT_INCOMPATIBLE_PARAMETER;
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
