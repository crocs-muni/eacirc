#include "XMLProcessor.h"
#include "generators/QuantumRndGen.h"
#include "generators/BiasRndGen.h"
#include <typeinfo>

void LoadConfigScript(TiXmlNode* pRoot, SETTINGS *pSettings) {
    // parsing EACIRC/NOTES
    pSettings->notes = getXMLElementValue(pRoot,"NOTES");

    // parsing EACIRC/MAIN
    pSettings->main.projectType = atoi(getXMLElementValue(pRoot,"MAIN/PROJECT").c_str());
    pSettings->main.evaluatorType = atoi(getXMLElementValue(pRoot,"MAIN/EVALUATOR").c_str());
    pSettings->main.evaluatorPrecision = atoi(getXMLElementValue(pRoot,"MAIN/EVALUATOR_PRECISION").c_str());
    pSettings->main.recommenceComputation = (atoi(getXMLElementValue(pRoot,"MAIN/RECOMMENCE_COMPUTATION").c_str())) ? true : false;
    pSettings->main.loadInitialPopulation = (atoi(getXMLElementValue(pRoot,"MAIN/LOAD_INITIAL_POPULATION").c_str())) ? true : false;
    pSettings->main.numGenerations = atol(getXMLElementValue(pRoot,"MAIN/NUM_GENERATIONS").c_str());
    pSettings->main.saveStateFrequency = atol(getXMLElementValue(pRoot,"MAIN/SAVE_STATE_FREQ").c_str());

    // parsing EACIRC/OUTPUTS
    pSettings->outputs.graphFiles = (atoi(getXMLElementValue(pRoot,"OUTPUTS/GRAPH_FILES").c_str())) ? true : false;
    pSettings->outputs.intermediateCircuits = (atoi(getXMLElementValue(pRoot,"OUTPUTS/INTERMEDIATE_CIRCUITS").c_str())) ? true : false;
    pSettings->outputs.allowPrunning = (atoi(getXMLElementValue(pRoot,"OUTPUTS/ALLOW_PRUNNING").c_str())) ? true : false;
    pSettings->outputs.saveTestVectors = (atoi(getXMLElementValue(pRoot,"OUTPUTS/SAVE_TEST_VECTORS").c_str())) ? true : false;

    // parsing EACIRC/RANDOM
    pSettings->random.useFixedSeed = (atoi(getXMLElementValue(pRoot,"RANDOM/USE_FIXED_SEED").c_str())) ? true : false;
    istringstream(getXMLElementValue(pRoot,"RANDOM/SEED")) >> pSettings->random.seed;
    pSettings->random.biasRndGenFactor = atoi(getXMLElementValue(pRoot,"RANDOM/BIAS_RNDGEN_FACTOR").c_str());
    pSettings->random.useNetShare = atoi(getXMLElementValue(pRoot,"RANDOM/USE_NET_SHARE").c_str()) ? true : false;
    pSettings->random.qrngPath = getXMLElementValue(pRoot,"RANDOM/QRNG_PATH");
    pSettings->random.qrngFilesMaxIndex = atoi(getXMLElementValue(pRoot,"RANDOM/QRNG_MAX_INDEX").c_str());

    // parsing EACIRC/CUDA
    pSettings->cuda.enabled = (atoi(getXMLElementValue(pRoot,"CUDA/ENABLED").c_str())) ? true : false;
    pSettings->cuda.something = getXMLElementValue(pRoot,"CUDA/SOMETHING");

    // parsing EACIRC/GA
    pSettings->ga.evolutionOff = atoi(getXMLElementValue(pRoot,"GA/EVOLUTION_OFF").c_str()) ? true : false;
    pSettings->ga.popupationSize = atoi(getXMLElementValue(pRoot,"GA/POPULATION_SIZE").c_str());
    pSettings->ga.replacementSize = atoi(getXMLElementValue(pRoot,"GA/REPLACEMENT_SIZE").c_str());
    pSettings->ga.probCrossing = (float) atof(getXMLElementValue(pRoot,"GA/PROB_CROSSING").c_str());
    pSettings->ga.probMutation = (float) atof(getXMLElementValue(pRoot,"GA/PROB_MUTATION").c_str());
    pSettings->ga.mutateFunctions = atoi(getXMLElementValue(pRoot,"GA/MUTATE_FUNCTIONS").c_str()) ? true : false;
    pSettings->ga.mutateConnectors = atoi(getXMLElementValue(pRoot,"GA/MUTATE_CONNECTORS").c_str()) ? true : false;

    // parsing EACIRC/CIRCUIT
    pSettings->circuit.numLayers = atoi(getXMLElementValue(pRoot,"CIRCUIT/NUM_LAYERS").c_str());
    pSettings->circuit.sizeLayer = atoi(getXMLElementValue(pRoot,"CIRCUIT/SIZE_LAYER").c_str());
    pSettings->circuit.sizeOutput = atoi(getXMLElementValue(pRoot,"CIRCUIT/SIZE_OUTPUT").c_str());
    pSettings->circuit.sizeInput = atoi(getXMLElementValue(pRoot,"CIRCUIT/SIZE_INPUT").c_str());
    pSettings->circuit.numConnectors = atoi(getXMLElementValue(pRoot,"CIRCUIT/NUM_CONNECTORS").c_str());
    pSettings->circuit.useMemory = atoi(getXMLElementValue(pRoot,"CIRCUIT/USE_MEMORY").c_str()) ? true : false;
    pSettings->circuit.sizeMemory = atoi(getXMLElementValue(pRoot,"CIRCUIT/SIZE_MEMORY").c_str());
    // parsing EACIRC/CIRCUIT/ALLOWED_FUNCTIONS
    pSettings->circuit.allowedFunctions[FNC_NOP] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_NOP").c_str());
    pSettings->circuit.allowedFunctions[FNC_CONS] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_CONS").c_str());
    pSettings->circuit.allowedFunctions[FNC_AND] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_AND").c_str());
    pSettings->circuit.allowedFunctions[FNC_NAND] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_NAND").c_str());
    pSettings->circuit.allowedFunctions[FNC_OR] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_OR").c_str());
    pSettings->circuit.allowedFunctions[FNC_XOR] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_XOR").c_str());
    pSettings->circuit.allowedFunctions[FNC_NOR] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_NOR").c_str());
    pSettings->circuit.allowedFunctions[FNC_NOT] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_NOT").c_str());
    pSettings->circuit.allowedFunctions[FNC_SHIL] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_SHIL").c_str());
    pSettings->circuit.allowedFunctions[FNC_SHIR] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_SHIR").c_str());
    pSettings->circuit.allowedFunctions[FNC_ROTL] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_ROTL").c_str());
    pSettings->circuit.allowedFunctions[FNC_ROTR] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_ROTR").c_str());
    pSettings->circuit.allowedFunctions[FNC_EQ] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_EQ").c_str());
    pSettings->circuit.allowedFunctions[FNC_LT] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_LT").c_str());
    pSettings->circuit.allowedFunctions[FNC_GT] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_GT").c_str());
    pSettings->circuit.allowedFunctions[FNC_LEQ] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_LEQ").c_str());
    pSettings->circuit.allowedFunctions[FNC_GEQ] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_GEQ").c_str());
    pSettings->circuit.allowedFunctions[FNC_BSLC] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_BSLC").c_str());
    pSettings->circuit.allowedFunctions[FNC_READ] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_READ").c_str());
    pSettings->circuit.allowedFunctions[FNC_EXT] = atoi(getXMLElementValue(pRoot,"CIRCUIT/ALLOWED_FUNCTIONS/FNC_EXT").c_str());

    // parsing EACIRC/POLYDIST
    pSettings->polydist.enabled = atoi(getXMLElementValue(pRoot,"POLYDIST/ENABLED").c_str()) ? true : false;
    pSettings->polydist.genomeInitMaxTerms = atoi(getXMLElementValue(pRoot,"POLYDIST/MAX_TERMS").c_str());
    pSettings->polydist.genomeInitTermCountProbability = atof(getXMLElementValue(pRoot,"POLYDIST/TERM_COUNT_P").c_str());
    pSettings->polydist.genomeInitTermStopProbability  = atof(getXMLElementValue(pRoot,"POLYDIST/TERM_VAR_P").c_str());
    pSettings->polydist.mutateAddTermProbability       = atof(getXMLElementValue(pRoot,"POLYDIST/ADD_TERM_P").c_str());
    pSettings->polydist.mutateAddTermStrategy          = atoi(getXMLElementValue(pRoot,"POLYDIST/ADD_TERM_STRATEGY").c_str());
    pSettings->polydist.mutateRemoveTermProbability    = atof(getXMLElementValue(pRoot,"POLYDIST/RM_TERM_P").c_str());
    pSettings->polydist.mutateRemoveTermStrategy       = atoi(getXMLElementValue(pRoot,"POLYDIST/RM_TERM_STRATEGY").c_str());
    pSettings->polydist.crossoverRandomizePolySelect   = atoi(getXMLElementValue(pRoot,"POLYDIST/CROSSOVER_RANDOMIZE_POLY").c_str()) ? true : false;
    pSettings->polydist.crossoverTermsProbability      = atoi(getXMLElementValue(pRoot,"POLYDIST/CROSSOVER_TERM_P").c_str()) ? true : false;
    
    // parsing EACIRC/TEST_VECTORS
    pSettings->testVectors.inputLength = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/INPUT_LENGTH").c_str());
    pSettings->testVectors.outputLength = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/OUTPUT_LENGTH").c_str());
    pSettings->testVectors.setSize = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/SET_SIZE").c_str());
    pSettings->testVectors.setChangeFrequency = atoi(getXMLElementValue(pRoot,"TEST_VECTORS/SET_CHANGE_FREQ").c_str());
    pSettings->testVectors.evaluateEveryStep = (atoi(getXMLElementValue(pRoot,"TEST_VECTORS/EVALUATE_EVERY_STEP").c_str())) ? true : false;
    pSettings->testVectors.evaluateBeforeTestVectorChange = (atoi(getXMLElementValue(pRoot,"TEST_VECTORS/EVALUATE_BEFORE_TEST_VECTOR_CHANGE").c_str())) ? true : false;

    // update extra info
    if (!pSettings->circuit.useMemory) {
        pSettings->circuit.sizeMemory = 0;
    }
    pSettings->testVectors.numTestSets = pSettings->main.numGenerations / pSettings->testVectors.setChangeFrequency;
    pSettings->circuit.sizeOutputLayer = pSettings->circuit.sizeOutput + pSettings->circuit.sizeMemory;
    pSettings->circuit.sizeInputLayer = pSettings->circuit.sizeInput + pSettings->circuit.sizeMemory;
    pSettings->circuit.genomeWidth = max(pSettings->circuit.sizeLayer, pSettings->circuit.sizeOutputLayer);
    // Compute genome size: genomeWidth for number of layers (each layer is twice - function and connector)
    pSettings->circuit.genomeSize = pSettings->circuit.numLayers * 2 * pSettings->circuit.genomeWidth;
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
        return pNode->ToElement()->GetText();
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
    return STAT_OK;
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
