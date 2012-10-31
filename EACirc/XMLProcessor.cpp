#include "XMLProcessor.h"
#include "random_generator/QuantumRndGen.h"
#include "random_generator/BiasRndGen.h"
#include <typeinfo>

int LoadConfigScript(string filePath, BASIC_INIT_DATA* pBasicSettings) {
    int status = STAT_OK;

    TiXmlNode* pRoot = NULL;
    loadXMLFile(pRoot, filePath);
    TiXmlHandle hRoot(pRoot);
    TiXmlElement* pElem;

    //
    //  PROGRAM VERSION AND DATE
    //
    pElem = hRoot.FirstChild("HEADER").FirstChildElement().Element();
    for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
        if (strcmp(pElem->Value(), "SWVERSION") == 0) pBasicSettings->simulSWVersion = pElem->GetText();
        if (strcmp(pElem->Value(), "SIMULDATE") == 0) pBasicSettings->simulDate = pElem->GetText();
    }

    //
    // RANDOM GENERATOR SETTINGS
    //
    pElem = hRoot.FirstChild("RANDOM_GENERATOR").FirstChildElement().Element();
    for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
        if (strcmp(pElem->Value(), "RANDOM_SEED") == 0) pBasicSettings->rndGen.randomSeed = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "USE_FIXED_SEED") == 0) pBasicSettings->rndGen.useFixedSeed = (atoi(pElem->GetText())) ? 1 : 0;
        if (strcmp(pElem->Value(), "QRGBS_PATH") == 0) pBasicSettings->rndGen.QRBGSPath = pElem->GetText();
        if (strcmp(pElem->Value(), "RNDGEN_TYPE") == 0) pBasicSettings->rndGen.type = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "BIAS_RNDGEN_BIAS_FACTOR") == 0) pBasicSettings->rndGen.biasFactor = atoi(pElem->GetText());
    }

    //
    // GA SETTINGS
    //
    pElem = hRoot.FirstChild("GA_CONFIG").FirstChildElement().Element();
    for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
        if (strcmp(pElem->Value(), "PROBABILITY_MUTATION") == 0) pBasicSettings->gaConfig.pMutt = atof(pElem->GetText());
        if (strcmp(pElem->Value(), "PROBABILITY_CROSSING") == 0) pBasicSettings->gaConfig.pCross = atof(pElem->GetText());
        if (strcmp(pElem->Value(), "POPULATION_SIZE") == 0) pBasicSettings->gaConfig.popSize = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "NUM_GENERATIONS") == 0) pBasicSettings->gaConfig.nGeners = atoi(pElem->GetText());
    }

    //
    // GA CIRCUIT SETTINGS
    //
    pElem = hRoot.FirstChild("GA_CIRCUIT_CONFIG").FirstChildElement().Element();
    for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
        if (strcmp(pElem->Value(), "TEST_VECTOR_LENGTH") == 0) pBasicSettings->gaCircuitConfig.testVectorLength = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "TEST_VECTOR_BALANCE") == 0) pBasicSettings->gaCircuitConfig.testVectorBalance = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "ALLOW_PRUNNING") == 0) pBasicSettings->gaCircuitConfig.allowPrunning = (atoi(pElem->GetText())) ? 1 : 0;
        if (strcmp(pElem->Value(), "NUM_LAYERS") == 0) pBasicSettings->gaCircuitConfig.numLayers = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "NUM_SELECTOR_LAYERS") == 0) pBasicSettings->gaCircuitConfig.numSelectorLayers = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "INPUT_LAYER_SIZE") == 0) pBasicSettings->gaCircuitConfig.numInputs = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "INTERNAL_LAYER_SIZE") == 0) pBasicSettings->gaCircuitConfig.internalLayerSize = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "OUTPUT_LAYER_SIZE") == 0) pBasicSettings->gaCircuitConfig.outputLayerSize = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "NUM_LAYER_CONNECTORS") == 0) pBasicSettings->gaCircuitConfig.numLayerConnectors = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "PREDICTION_METHOD") == 0) pBasicSettings->gaCircuitConfig.predictMethod = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "GENOME_SIZE") == 0) pBasicSettings->gaCircuitConfig.genomeSize = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "NUM_TEST_VECTORS") == 0) pBasicSettings->gaCircuitConfig.numTestVectors = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "GENER_CHANGE_SEED") == 0) pBasicSettings->gaCircuitConfig.testVectorGenerChangeSeed = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "SAVE_TEST_VECTORS") == 0) pBasicSettings->gaCircuitConfig.saveTestVectors = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "TEST_VECTOR_CHANGE_GENERATION") == 0) pBasicSettings->gaCircuitConfig.testVectorChangeGener = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "TVCG_PROGRESSIVE") == 0) pBasicSettings->gaCircuitConfig.TVCGProgressive = (atoi(pElem->GetText())) ? 1 : 0;
        if (strcmp(pElem->Value(), "EVALUATE_EVERY_STEP") == 0) pBasicSettings->gaCircuitConfig.evaluateEveryStep = (atoi(pElem->GetText())) ? 1 : 0;
        if (strcmp(pElem->Value(), "TEST_VECTOR_GENERATION_METHOD") == 0) pBasicSettings->gaCircuitConfig.testVectorGenerMethod = atoi(pElem->GetText());
    }

    //
    // ESTREAM TEST VECTOR CONFIG (IF ENABLED)
    //
    if (pBasicSettings->gaCircuitConfig.testVectorGenerMethod == ESTREAM_CONST) {
        pElem = hRoot.FirstChild("GA_CIRCUIT_CONFIG").FirstChild("ESTREAM_SETTINGS").FirstChildElement().Element();
        for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
            if (strcmp(pElem->Value(), "ESTREAM_GENERATION_METHOD") == 0) pBasicSettings->gaCircuitConfig.testVectorEstreamMethod = atoi(pElem->GetText());
            if (strcmp(pElem->Value(), "LIMIT_ALG_ROUNDS") == 0) pBasicSettings->gaCircuitConfig.limitAlgRounds = (atoi(pElem->GetText())) ? 1 : 0;
            if (strcmp(pElem->Value(), "NUM_ALG_ROUNDS") == 0) pBasicSettings->gaCircuitConfig.limitAlgRoundsCount = atoi(pElem->GetText());
            if (strcmp(pElem->Value(), "NUM_ALG_ROUNDS2") == 0) pBasicSettings->gaCircuitConfig.limitAlgRoundsCount2 = atoi(pElem->GetText());
            if (strcmp(pElem->Value(), "ESTREAM_ALGORITHM") == 0) pBasicSettings->gaCircuitConfig.testVectorEstream = atoi(pElem->GetText());
            if (strcmp(pElem->Value(), "ESTREAM_ALGORITHM2") == 0) pBasicSettings->gaCircuitConfig.testVectorEstream2 = atoi(pElem->GetText());
            if (strcmp(pElem->Value(), "ESTREAM_INPUTTYPE") == 0) pBasicSettings->gaCircuitConfig.estreamInputType= atoi(pElem->GetText());
            if (strcmp(pElem->Value(), "ESTREAM_KEYTYPE") == 0) pBasicSettings->gaCircuitConfig.estreamKeyType = atoi(pElem->GetText());
            if (strcmp(pElem->Value(), "ESTREAM_IVTYPE") == 0) pBasicSettings->gaCircuitConfig.estreamIVType = atoi(pElem->GetText());
        }
    }

    //
    // ALLOWED FUNCTIONS IN CIRCUIIT
    //
    pElem = hRoot.FirstChild("GA_CIRCUIT_CONFIG").FirstChild("ALLOWED_FNC").FirstChildElement().Element();
    for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
        if (strcmp(pElem->Value(), "FNC_NOP") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_NOP] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_OR") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_OR] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_AND") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_AND] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_CONST") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_CONST] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_XOR") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_XOR] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_NOR") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_NOR] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_NAND") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_NAND] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_ROTL") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_ROTL] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_ROTR") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_ROTR] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_SUM") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_SUM] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_SUBS") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_SUBS] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_ADD") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_ADD] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_MULT") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_MULT] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_DIV") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_DIV] = atoi(pElem->GetText());
        if (strcmp(pElem->Value(), "FNC_READX") == 0) pBasicSettings->gaCircuitConfig.allowedFNC[FNC_READX] = atoi(pElem->GetText());
    }

    delete pRoot;
    return status;
}

int saveXMLFile(TiXmlNode* pRoot, string filename) {
    TiXmlDocument doc;
    TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );
    doc.LinkEndChild(decl);
    doc.LinkEndChild(pRoot);
    bool result = doc.SaveFile(filename.c_str());
    if (!result) {
        mainLogger.out() << "Error: cannot write XML file " << filename << ".";
        return STAT_FILE_OPEN_FAIL;
    }
    return STAT_OK;
}

int loadXMLFile(TiXmlNode*& pRoot, string filename) {
    TiXmlDocument doc(filename.c_str());
    if (!doc.LoadFile()) {
        mainLogger.out() << "Error: Could not load file '" << filename << "'." << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    TiXmlHandle hDoc(&doc);
    TiXmlElement* pElem=hDoc.FirstChildElement().Element();
    if (!pElem) {
        mainLogger.out() << "Error: No root element in XML (" << filename << ")." << endl;
        return STAT_FILE_OPEN_FAIL;
    }
    pRoot = pElem->Clone();

    return STAT_OK;
}
