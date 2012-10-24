#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "math.h"
#include "time.h"

#include "EACirc.h"
#include "EACglobals.h"
//#include "globals.h"
#include "CommonFnc.h"
#include "random_generator/IRndGen.h"
#include "random_generator/BiasRndGen.h"
//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
//libinclude (tinyXML/tinyxml.h)
#include "tinyxml.h"
#include "Evaluator.h"
#include "CircuitGenome.h"
#include "estream/estreamInterface.h"
#include "test_vector_generator/ITestVectGener.h"
#include "test_vector_generator/EstreamVectGener.h"
#include "estream/EncryptorDecryptor.h"
#include "EAC_circuit.h"
#include "standalone_testers/TestDistinctorCircuit.h"

#ifdef _WIN32
	#include <Windows.h>
	#define getpid() GetCurrentProcessId()
#endif
#ifdef __linux__
	#include <sys/types.h>
	#include <unistd.h>
#endif

IRndGen*                rndGen;
IRndGen*                biasRndGen;
GA_CIRCUIT*             pGACirc = NULL;
EncryptorDecryptor*		encryptorDecryptor = NULL;
Logger                  mainLogger;

int LoadConfigScript(string filePath, BASIC_INIT_DATA* pBasicSettings) {
    int     status = STAT_OK;
	string	value;

	TiXmlDocument doc(filePath.c_str());
	if (!doc.LoadFile())
        return STAT_CONFIG_DATA_READ_FAIL;

	TiXmlElement* pElem;
	TiXmlHandle hRoot(&doc);
	
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

    return status;
}

int main(int argc, char **argv)
{

	int status = STAT_OK;
    //int resumeStatus = STAT_FILE_OPEN_FAIL;
	unsigned long seed = 0;
	BASIC_INIT_DATA pBasicSettings;

    status = LoadConfigScript(FILE_CONFIG, &pBasicSettings);
    if (status == STAT_CONFIG_DATA_READ_FAIL) {
        mainLogger.out() << "Could not read configuration data from " << FILE_CONFIG << endl;
        return status;
    }

	// CREATE STRUCTURE OF CIRCUIT FROM BASIC SETTINGS
	pGACirc = &(pBasicSettings.gaCircuitConfig);
	pGACirc->allocate();

    //
    // COMMAND LINE ARGUMENTS PROCESSING
    //
    bool evolutionOff = false;
    if (argc > 1) {
        int i = 0;
        while (++i < argc) {
            if (strcmp(argv[i],CMD_OPT_LOGGING) == 0) {
                mainLogger.setOutputStream();
                mainLogger.setlogging(true);
            } else
            if (strcmp(argv[i],CMD_OPT_LOGGING_TO_FILE) == 0) {
                mainLogger.setOutputFile();
                mainLogger.setlogging(true);
            } else
              // STATIC CIRCUIT ?
            if (strcmp(argv[i],CMD_OPT_STATIC) == 0) {
                if (argc >= i && strcmp(argv[i+1],CMD_OPT_STATIC_DISTINCTOR) == 0) {
                    mainLogger.out() << "Static circuit, distinctor mode." << endl;
                    return testDistinctorCircuit(string(FILE_TEST_DATA_1), string(FILE_TEST_DATA_2));
                } else {
                    mainLogger.out() << "Please specify the second parameter. Supported options:" << endl;
                    mainLogger.out() << "  " << CMD_OPT_STATIC_DISTINCTOR << "  (use the circuit as distinctor)" << endl;
                    return STAT_INVALID_ARGUMETS;
                }
            } else
              // EVOLUTION IS OFF ?
            if (strcmp(argv[i],CMD_OPT_EVOLUTION_OFF) == 0) {
                evolutionOff = true;
                mainLogger.out() << "Evolution turned off." << endl;
            } else {
                mainLogger.out() << "\"" << argv[1] << "\" is not a valid argument." << endl;
                mainLogger.out() << "Only valid arguments for EACirc are:" << endl;
                mainLogger.out() << "  " << CMD_OPT_STATIC << "  (run tests on precompiled circuit)" << endl;
                mainLogger.out() << "  " << CMD_OPT_EVOLUTION_OFF << "  (do not evolve circuits)" << endl;
                return STAT_INVALID_ARGUMETS;
            }
        }
    }

	// PREPARE THE LOGGING FILES
    std::remove(FILE_FITNESS_PROGRESS);
    std::remove(FILE_BEST_FITNESS);
    std::remove(FILE_AVG_FITNESS);

	// RESTORE THE SEED
	fstream	sfile;
	string sseed;
	
	//with useFixedSeed, a seed file is used, upon fail, randomseed argument is used
	if (pBasicSettings.rndGen.useFixedSeed) {
		if (!sfile.is_open())
            sfile.open(FILE_SEEDFILE, fstream::in);
		getline(sfile, sseed);
        // why?
		cout << sseed << endl;
		if (!sseed.empty())
			seed = atoi(sseed.c_str());
        sfile.close();

		// USE STATIC SEED
        if (!seed) {
            seed = pBasicSettings.rndGen.randomSeed;
            mainLogger.out() << "Using fixed seed: " << seed << endl;
        }
	}

	srand(clock() + time(NULL) + getpid());
	//srand((unsigned int) time(NULL));
	if (seed == 0){
		seed = (rand() %100000) + ((rand() %42946) *100000);
        mainLogger.out() << "Using system-generated random seed: " << seed << endl;
	}
	
	//INIT RNG
	GARandomSeed(seed);
	rndGen = new IRndGen(pBasicSettings.rndGen.type);
    //rndGen = rndGen->getRndGenClass();
    rndGen = rndGen->getInitializedRndGenClass(seed,pBasicSettings.rndGen.QRBGSPath);
    //mainLogger.out() << "deterministic from now on" << endl;
    //rndGen->InitRandomGenerator(seed,pBasicSettings.rndGen.QRBGSPath);
    mainLogger.out() << "Random generator initialized (" << rndGen->ToString() << ")" <<endl;

	//INIT BIAS RNDGEN
	biasRndGen = new IRndGen(BIASGEN);
    //biasRndGen = biasRndGen->getRndGenClass();
    biasRndGen = biasRndGen->getInitializedRndGenClass(seed,pBasicSettings.rndGen.QRBGSPath);
	((BiasRndGen*)biasRndGen)->setChanceForOne(pBasicSettings.rndGen.biasFactor);
    //biasRndGen->InitRandomGenerator(seed,pBasicSettings.rndGen.QRBGSPath);
    mainLogger.out() << "Bias random generator initialized (" << biasRndGen->ToString() << ")" <<endl;

	//INIT ENCRYPTOR/DECRYPTOR
	encryptorDecryptor = new EncryptorDecryptor();

	//LOG THE SEED
    ofstream ssfile(FILE_SEEDFILE, ios::app);
	ssfile << "----------";
	if (pBasicSettings.rndGen.useFixedSeed)
		ssfile << "Using fixed seed" << endl;
	else
		ssfile << "Using random seed" << endl;
	ssfile << seed << endl;
	ssfile.close();

	//LOG THE TESTVECTGENER METHOD
	if (pGACirc->testVectorGenerMethod == ESTREAM_CONST) {
        ofstream out(FILE_FITNESS_PROGRESS, ios::app);
		out << "Using Ecrypt candidate n." << pGACirc->testVectorEstream << " (" <<  pBasicSettings.gaCircuitConfig.limitAlgRoundsCount << " rounds) AND candidate n." << pGACirc->testVectorEstream2 << " (" << pBasicSettings.gaCircuitConfig.limitAlgRoundsCount2 << " rounds)" <<  endl;
		out.close();
        mainLogger.out() << "stream1: using " << estreamToString(pGACirc->testVectorEstream);
        mainLogger.out() << " (" << pBasicSettings.gaCircuitConfig.limitAlgRoundsCount << " rounds)" << endl;
        mainLogger.out() << "stream2: using " << estreamToString(pGACirc->testVectorEstream2);
        mainLogger.out() << " (" << pBasicSettings.gaCircuitConfig.limitAlgRoundsCount2 << " rounds)" << endl;
	}

    GA1DArrayGenome<unsigned long> genom(pGACirc->genomeSize, CircuitGenome::Evaluator);

	// LOAD genome
	string fileName = "EAC_circuit.bin";
	fstream	efile;
	string executetext;
	efile.open(fileName.c_str(), fstream::in);
	
	if (efile.is_open()) {
        mainLogger.out() << "Loading genome from file." << endl;
		getline(efile, executetext);
		CircuitGenome::ExecuteFromText(executetext, &genom);
        efile.close();
	}

    if (status == STAT_OK) {
        ofstream out(FILE_FITNESS_PROGRESS, ios::app);
		
        // INIT EVALUATOR
		Evaluator *evaluator = new Evaluator();

		out << "evalOK" << endl;
		//  CREATE GA STRUCTS
		GA1DArrayGenome<unsigned long> genomeTemp(pGACirc->genomeSize, CircuitGenome::Evaluator);
		
		genom.initializer(CircuitGenome::Initializer);
		genom.mutator(CircuitGenome::Mutator);
		genom.crossover(CircuitGenome::Crossover);

		GASteadyStateGA ga(genom);
		
		ga.populationSize(pBasicSettings.gaConfig.popSize);
		ga.nReplacement(2 * pBasicSettings.gaConfig.popSize / 3);
		ga.nGenerations(pBasicSettings.gaConfig.nGeners);
		ga.pCrossover(pBasicSettings.gaConfig.pCross);
		ga.pMutation(pBasicSettings.gaConfig.pMutt);
        ga.scoreFilename(FILE_GALIB_SCORES);
		ga.scoreFrequency(1);	// keep the scores of every generation
		ga.flushFrequency(1);	// specify how often to write the score to disk 
		ga.selectScores(GAStatistics::AllScores);

		ga.initialize();

		out << "GAOK" << endl;
        mainLogger.out() << "GAlib fully initialized." << endl;
		int		actGener = 1;
		int		changed = 1;
		int		evaluateNext = 0;
		
		pGACirc->clearFitnessStats();

		fstream fitfile;

        if (evolutionOff) {
			genomeTemp = genom;
        }
		
        while (actGener < pBasicSettings.gaConfig.nGeners) {
			//FRACTION FILE FOR BOINC
            fitfile.open(FILE_BOINC_FRACTION_DONE, fstream::out | ios::trunc);
			fitfile << ((float)(actGener))/((float)(pBasicSettings.gaConfig.nGeners));
			fitfile.close();
            
			// DO NOT EVOLVE..
            if (evolutionOff) {
				evaluator->generateTestVectors();
				evaluator->evaluateStep(genom, actGener);
				actGener++;
			} else {
				// RESET EVALUTION FOR ALL GENOMS
				ga.pop->flushEvalution();
				ga.step(); // GA evolution step

				if (evaluateNext) {
					evaluator->evaluateStep(genomeTemp, actGener);
					evaluateNext = 0;
				}

				genomeTemp = (GA1DArrayGenome<unsigned long>&) ga.population().best();// .statistics().bestIndividual();
            
				if ((pGACirc->TVCGProgressive && (changed > actGener/pGACirc->testVectorChangeGener + 1)) ||
					(!pGACirc->TVCGProgressive && ((actGener %(pGACirc->testVectorChangeGener)) == 0))) {

					if (pGACirc->testVectorGenerChangeSeed == 1) {
						//set a new seed
                        ofstream ssfile(FILE_SEEDFILE, ios::app);

                        rndGen->GetRandomFromInterval(4294967295, &seed);
						GARandomSeed(seed);
						rndGen->InitRandomGenerator(seed,pBasicSettings.rndGen.QRBGSPath);
						ssfile << GAGetRandomSeed() << endl;
						ssfile.close();
					}

					// GENERATE FRESH SET AND EVALUATE ONLY THE BEST ONE
					evaluator->generateTestVectors();
					evaluateNext = 1;
					changed = 0;
				}
				else if (pGACirc->evaluateEveryStep) evaluator->evaluateStep(genomeTemp, actGener);
				changed++;
				actGener++;
			}
		}
		// GENERATE FRESH NEW SET AND EVALUATE THE RESULT
		evaluator->generateTestVectors();
		evaluator->evaluateStep(genomeTemp, actGener);

		//Print the circuit
		CircuitGenome::PrintCircuit(genomeTemp,"",0,1);
    }   

    return status;
}
