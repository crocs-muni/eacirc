#include "XMLProcessor.h"
#include "random_generator/QuantumRndGen.h"
#include "random_generator/BiasRndGen.h"
#include <typeinfo>

int LoadConfigScript(string filePath, BASIC_INIT_DATA* pBasicSettings) {
    int status = STAT_OK;

    TiXmlNode* pRoot = NULL;
    status = loadXMLFile(pRoot, filePath);
    TiXmlHandle hRoot(pRoot);
    TiXmlElement* pElem;

    //
    // PROGRAM VERSION AND DATE
    //
    pElem = hRoot.FirstChild("INFO").FirstChildElement().Element();
    for( pElem; pElem; pElem = pElem->NextSiblingElement()) {
        if (string(pElem->Value()) == "DATE") pBasicSettings->simulDate = pElem->GetText();
        if (string(pElem->Value()) == "VERSION") pBasicSettings->simulSWVersion = pElem->GetText();
    }

    //
    // MAIN SETTINGS
    //
    pElem = hRoot.FirstChild("MAIN").FirstChildElement().Element();
    for (pElem; pElem; pElem = pElem->NextSiblingElement()) {
        if (string(pElem->Value()) == "LOAD_STATE")
            pBasicSettings->loadState = (atoi(pElem->GetText())) ? true : false;
        if (string(pElem->Value()) == "NUM_GENERATIONS")
            pBasicSettings->gaConfig.nGeners = atol(pElem->GetText());
        if (string(pElem->Value()) == "SAVE_STATE_FREQ")
            pBasicSettings->gaCircuitConfig.changeGalibSeedFrequency = atol(pElem->GetText());
        if (string(pElem->Value()) == "TEST_VECTOR_GENERATION_METHOD")
            pBasicSettings->gaCircuitConfig.testVectorGenerMethod = atol(pElem->GetText());
        if (string(pElem->Value()) == "QRNG_PATH")
            pBasicSettings->rndGen.QRBGSPath = pElem->GetText();
    }

    //
    // RANDOM GENERATOR SETTINGS
    //
    pElem = hRoot.FirstChild("RANDOM").FirstChildElement().Element();
    for( pElem; pElem; pElem = pElem->NextSiblingElement()) {
        if (string(pElem->Value()) == "USE_FIXED_SEED")
            pBasicSettings->rndGen.useFixedSeed = (atoi(pElem->GetText())) ? true : false;
        if (string(pElem->Value()) == "SEED") {
            istringstream textSeed(pElem->GetText());
            unsigned long seed;
            textSeed >> seed;
            pBasicSettings->rndGen.randomSeed = seed;
        }
        if (string(pElem->Value()) == "PRIMARY_RANDOM_TYPE")
            pBasicSettings->rndGen.type = atoi(pElem->GetText());
        if (string(pElem->Value()) == "BIAS_RNDGEN_FACTOR")
            pBasicSettings->rndGen.biasFactor = atoi(pElem->GetText());
    }

    //
    // GA SETTINGS
    //
    pElem = hRoot.FirstChild("GA").FirstChildElement().Element();
    for( pElem; pElem; pElem = pElem->NextSiblingElement()) {
        if (string(pElem->Value()) == "PROB_MUTATION")
            pBasicSettings->gaConfig.pMutt = atof(pElem->GetText());
        if (string(pElem->Value()) == "PROB_CROSSING")
            pBasicSettings->gaConfig.pCross = atof(pElem->GetText());
        if (string(pElem->Value()) == "POPULATION_SIZE")
            pBasicSettings->gaConfig.popSize = atoi(pElem->GetText());
    }

    //
    // GA CIRCUIT SETTINGS
    //
    pElem = hRoot.FirstChild("CIRCUIT").FirstChildElement().Element();
    for( pElem; pElem; pElem = pElem->NextSiblingElement()) {
        if (string(pElem->Value()) == "NUM_LAYERS")
            pBasicSettings->gaCircuitConfig.numLayers = atoi(pElem->GetText());
        if (string(pElem->Value()) == "NUM_SELECT_LAYERS")
            pBasicSettings->gaCircuitConfig.numSelectorLayers = atoi(pElem->GetText());
        if (string(pElem->Value()) == "SIZE_LAYER")
            pBasicSettings->gaCircuitConfig.internalLayerSize = atoi(pElem->GetText());
        if (string(pElem->Value()) == "SIZE_OUTPUT_LAYER")
            pBasicSettings->gaCircuitConfig.outputLayerSize = atoi(pElem->GetText());
        if (string(pElem->Value()) == "SIZE_INPUT_LAYER")
            pBasicSettings->gaCircuitConfig.numInputs = atoi(pElem->GetText());
        if (string(pElem->Value()) == "NUM_CONNECTORS")
            pBasicSettings->gaCircuitConfig.numLayerConnectors = atoi(pElem->GetText());
        if (string(pElem->Value()) == "PREDICTION_METHOD")
            pBasicSettings->gaCircuitConfig.predictMethod = atoi(pElem->GetText());
        if (string(pElem->Value()) == "GENOME_SIZE")
            pBasicSettings->gaCircuitConfig.genomeSize = atoi(pElem->GetText());
        if (string(pElem->Value()) == "ALLOW_PRUNNING")
            pBasicSettings->gaCircuitConfig.allowPrunning = (atoi(pElem->GetText())) ? true : false;
        if (string(pElem->Value()) == "ALLOWED_FUNCTIONS") {
            for(TiXmlElement* pElem2 = pElem; pElem2; pElem2=pElem2->NextSiblingElement()) {
                if (string(pElem2->Value()) == "FNC_NOP") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_NOP] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_OR") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_OR] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_AND") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_AND] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_CONST") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_CONST] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_XOR") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_XOR] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_NOR") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_NOR] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_NAND") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_NAND] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_ROTL") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_ROTL] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_ROTR") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_ROTR] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_SUM") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_SUM] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_SUBS") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_SUBS] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_ADD") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_ADD] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_MULT") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_MULT] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_DIV") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_DIV] = atoi(pElem2->GetText());
                if (string(pElem2->Value()) == "FNC_READX") pBasicSettings->gaCircuitConfig.allowedFNC[FNC_READX] = atoi(pElem2->GetText());
            }
        }
    }

    //
    // TEST VECTORS
    //
    pElem = hRoot.FirstChild("TEST_VECTORS").FirstChildElement().Element();
    for( pElem; pElem; pElem = pElem->NextSiblingElement()) {
        if (string(pElem->Value()) == "TEST_VECTOR_LENGTH")
            pBasicSettings->gaCircuitConfig.testVectorLength = atoi(pElem->GetText());
        if (string(pElem->Value()) == "NUM_TEST_VECTORS")
            pBasicSettings->gaCircuitConfig.numTestVectors = atoi(pElem->GetText());
        if (string(pElem->Value()) == "TEST_VECTOR_CHANGE_FREQ")
            pBasicSettings->gaCircuitConfig.testVectorChangeGener = atoi(pElem->GetText());
        if (string(pElem->Value()) == "TEST_VECTOR_CHANGE_PROGRESSIVE")
            pBasicSettings->gaCircuitConfig.TVCGProgressive = (atoi(pElem->GetText())) ? true : false;
        if (string(pElem->Value()) == "SAVE_TEST_VECTORS")
            pBasicSettings->gaCircuitConfig.saveTestVectors = atoi(pElem->GetText());
        if (string(pElem->Value()) == "BALLANCED_TEST_VECTORS")
            pBasicSettings->gaCircuitConfig.testVectorBalance = atoi(pElem->GetText());
        if (string(pElem->Value()) == "EVALUATE_EVERY_STEP")
            pBasicSettings->gaCircuitConfig.evaluateEveryStep = (atoi(pElem->GetText())) ? true : false;
    }

    //
    // ESTREAM TEST VECTOR CONFIG (IF ENABLED)
    //
    if (pBasicSettings->gaCircuitConfig.testVectorGenerMethod == ESTREAM_CONST) {
        pElem = hRoot.FirstChild("ESTREAM").FirstChildElement().Element();
        for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
            if (string(pElem->Value()) == "ESTREAM_USAGE_TYPE")
                pBasicSettings->gaCircuitConfig.testVectorEstreamMethod = atoi(pElem->GetText());
            if (string(pElem->Value()) == "ALGORITHM_1")
                pBasicSettings->gaCircuitConfig.testVectorEstream = atoi(pElem->GetText());
            if (string(pElem->Value()) == "ALGORITHM_2")
                pBasicSettings->gaCircuitConfig.testVectorEstream2 = atoi(pElem->GetText());
            if (string(pElem->Value()) == "LIMIT_NUM_OF_ROUNDS")
                pBasicSettings->gaCircuitConfig.limitAlgRounds = (atoi(pElem->GetText())) ? true : false;
            if (string(pElem->Value()) == "ROUNDS_ALG_1")
                pBasicSettings->gaCircuitConfig.limitAlgRoundsCount = atoi(pElem->GetText());
            if (string(pElem->Value()) == "ROUNDS_ALG_2")
                pBasicSettings->gaCircuitConfig.limitAlgRoundsCount2 = atoi(pElem->GetText());
            if (string(pElem->Value()) == "PLAINTEXT_TYPE")
                pBasicSettings->gaCircuitConfig.estreamInputType= atoi(pElem->GetText());
            if (string(pElem->Value()) == "KEY_TYPE")
                pBasicSettings->gaCircuitConfig.estreamKeyType = atoi(pElem->GetText());
            if (string(pElem->Value()) == "IV_TYPE")
                pBasicSettings->gaCircuitConfig.estreamIVType = atoi(pElem->GetText());
        }
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
