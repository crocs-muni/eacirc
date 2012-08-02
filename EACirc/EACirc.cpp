#include "SSGlobals.h"
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "globals.h"
#include "EACirc.h"
#include "CommonFnc.h"
#include "status.h"
#include "Random Generator/IRndGen.h"
#include "Random Generator/BiasRndGen.h"
//libinclude (ga/GA1DArrayGenome.h)
#include "ga/GA1DArrayGenome.h"
//libinclude (tinyXML/tinyxml.h)
#include "tinyXML/tinyxml.h"
#include "math.h"
#include "time.h"
#include "Evaluator.h"
#include "CircuitGenome.h"
#include "estream-interface.h"
#include "ITestVectGener.h"
#include "EstreamVectGener.h"
#include "EncryptorDecryptor.h"
#include "EAC_circuit.h"
#include "Standalone testers/TestDistinctorCircuit.h"

#ifdef _WIN32
	#include <Windows.h>
	#define getpid() GetCurrentProcessId()
#endif
#ifdef __linux__
	#include <sys/types.h>
	#include <unistd.h>
#endif

IRndGen*         rndGen;
IRndGen*         biasRndGen;
GA_CIRCUIT*		pGACirc = NULL;
EncryptorDecryptor*		encryptorDecryptor = NULL;

int LoadConfigScript(string filePath, BASIC_INIT_DATA* pBasicSettings) {
    int     status = STAT_OK;
	string	value;

	TiXmlDocument doc(filePath.c_str());
	if (!doc.LoadFile())
		return STAT_INI_DATA_READ_FAIL;

	TiXmlElement* pElem;
	TiXmlHandle hRoot(&doc);
	
    //
    //  PROGRAM VERSION AND DATE
    //
	pElem = hRoot.FirstChild("HEADER").FirstChild().Element();
	for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
		if (strcmp(pElem->Value(), "SWVERSION") == 0) pBasicSettings->simulSWVersion = pElem->GetText();
		if (strcmp(pElem->Value(), "SIMULDATE") == 0) pBasicSettings->simulDate = pElem->GetText();
	}

	//
	// RANDOM GENERATOR SETTINGS
	//
	pElem = hRoot.FirstChild("RANDOM_GENERATOR").FirstChild().Element();
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
	pElem = hRoot.FirstChild("GA_CONFIG").FirstChild().Element();
	for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
		if (strcmp(pElem->Value(), "PROBABILITY_MUTATION") == 0) pBasicSettings->gaConfig.pMutt = atof(pElem->GetText());
		if (strcmp(pElem->Value(), "PROBABILITY_CROSSING") == 0) pBasicSettings->gaConfig.pCross = atof(pElem->GetText());
		if (strcmp(pElem->Value(), "POPULATION_SIZE") == 0) pBasicSettings->gaConfig.popSize = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "NUM_GENERATIONS") == 0) pBasicSettings->gaConfig.nGeners = atoi(pElem->GetText());
	}

	//
	// GA CIRCUIT SETTINGS
	//
	pElem = hRoot.FirstChild("GA_CIRCUIT_CONFIG").FirstChild().Element();
	for( pElem; pElem; pElem=pElem->NextSiblingElement()) {
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
		if (strcmp(pElem->Value(), "LIMIT_ALG_ROUNDS") == 0) pBasicSettings->gaCircuitConfig.limitAlgRounds = (atoi(pElem->GetText())) ? 1 : 0;
		if (strcmp(pElem->Value(), "NUM_ALG_ROUNDS") == 0) pBasicSettings->gaCircuitConfig.limitAlgRoundsCount = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "GENER_CHANGE_SEED") == 0) pBasicSettings->gaCircuitConfig.testVectorGenerChangeSeed = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "TEST_VECTOR_GENERATION_METHOD") == 0) pBasicSettings->gaCircuitConfig.testVectorGenerMethod = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "TEST_VECTOR_LENGTH") == 0) pBasicSettings->gaCircuitConfig.testVectorLength = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "TEST_VECTOR_BALANCE") == 0) pBasicSettings->gaCircuitConfig.testVectorBalance = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "ESTREAM_ALGORITHM") == 0) pBasicSettings->gaCircuitConfig.testVectorEstream = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "ESTREAM_ALGORITHM2") == 0) pBasicSettings->gaCircuitConfig.testVectorEstream2 = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "ESTREAM_INPUTTYPE") == 0) pBasicSettings->gaCircuitConfig.estreamInputType= atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "ESTREAM_KEYTYPE") == 0) pBasicSettings->gaCircuitConfig.estreamKeyType = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "ESTREAM_IVTYPE") == 0) pBasicSettings->gaCircuitConfig.estreamIVType = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "NUM_ALG_ROUNDS2") == 0) pBasicSettings->gaCircuitConfig.limitAlgRoundsCount2 = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "SAVE_TEST_VECTORS") == 0) pBasicSettings->gaCircuitConfig.saveTestVectors = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "ESTREAM_GENERATION_METHOD") == 0) pBasicSettings->gaCircuitConfig.testVectorEstreamMethod = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "TEST_VECTOR_CHANGE_GENERATION") == 0) pBasicSettings->gaCircuitConfig.testVectorChangeGener = atoi(pElem->GetText());
		if (strcmp(pElem->Value(), "TVCG_PROGRESSIVE") == 0) pBasicSettings->gaCircuitConfig.TVCGProgressive = (atoi(pElem->GetText())) ? 1 : 0;
		if (strcmp(pElem->Value(), "EVALUATE_EVERY_STEP") == 0) pBasicSettings->gaCircuitConfig.evaluateEveryStep = (atoi(pElem->GetText())) ? 1 : 0;
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
	int resumeStatus = STAT_FILE_OPEN_FAIL;
	unsigned long seed = 0;
	string seedFile = "LastSeed.txt";
	string filePath = "config.xml";
	BASIC_INIT_DATA pBasicSettings;

	status = LoadConfigScript(filePath, &pBasicSettings);

	// CREATE STRUCTURE OF CIRCUIT FROM BASIC SETTINGS
	pGACirc = &(pBasicSettings.gaCircuitConfig);
	pGACirc->allocate();

	// STATIC CIRCUIT TESTING
	if (argc>1 && strcmp(argv[1],"-staticcircuit") == 0) {
		if (argc == 2) {
			cout << "Please specify the second parameter:" << endl;
			cout << "-distinctor		- use the circuit as distinctor" << endl;
			return 0;
		}
		if (strcmp(argv[2],"-distinctor") == 0)
			return testDistinctorCircuit(string("TestData1.txt"), string("TestData2.txt"));
	}

	// PREPARE THE LOGGING FILES
    std::remove("EAC_fitnessProgress.txt");
	std::remove("bestfitgraph.txt");
	std::remove("avgfitgraph.txt");

	// RESTORE THE SEED
	fstream	sfile;
	string sseed;
	
	//with useFixedSeed, a seed file is used, upon fail, randomseed argument is used
	if (pBasicSettings.rndGen.useFixedSeed) {
		if (!sfile.is_open())
			sfile.open(seedFile.c_str(), fstream::in);
		getline(sfile, sseed);
		cout << sseed << endl;
		if (!sseed.empty())
			seed = atoi(sseed.c_str());
		sfile.close();

		// USE STATIC SEED
		if (!seed) seed = pBasicSettings.rndGen.randomSeed;
	}

	srand(clock() + time(NULL) + getpid());
	//srand((unsigned int) time(NULL));
	if (seed == 0){
		seed = (rand() %100000) + ((rand() %42946) *100000);
	}
	
	//INIT RNG
	GARandomSeed(seed);
	rndGen = new IRndGen(pBasicSettings.rndGen.type);
	rndGen = rndGen->getRndGenClass();
	rndGen->InitRandomGenerator(seed,pBasicSettings.rndGen.QRBGSPath);

	//INIT BIAS RNDGEN
	biasRndGen = new IRndGen(BIASGEN);
	biasRndGen = biasRndGen->getRndGenClass();
	((BiasRndGen*)biasRndGen)->setChanceForOne(pBasicSettings.rndGen.biasFactor);
	biasRndGen->InitRandomGenerator(seed,pBasicSettings.rndGen.QRBGSPath);

	//INIT ENCRYPTOR/DECRYPTOR
	encryptorDecryptor = new EncryptorDecryptor();

	//LOG THE SEED
	ofstream ssfile(seedFile.c_str(), ios::app);
	ssfile << "----------";
	if (pBasicSettings.rndGen.useFixedSeed)
		ssfile << "Using fixed seed" << endl;
	else
		ssfile << "Using random seed" << endl;
	ssfile << seed << endl;
	ssfile.close();

	//LOG THE TESTVECTGENER METHOD
	if (pGACirc->testVectorGenerMethod == ESTREAM_CONST) {
		ofstream out("EAC_fitnessProgress.txt", ios::app);
		out << "Using Ecrypt candidate n." << pGACirc->testVectorEstream << " (" <<  pBasicSettings.gaCircuitConfig.limitAlgRoundsCount << " rounds) AND candidate n." << pGACirc->testVectorEstream2 << " (" << pBasicSettings.gaCircuitConfig.limitAlgRoundsCount2 << " rounds)" <<  endl;
		out.close();
	}

	GA1DArrayGenome<unsigned long>    genom(pGACirc->genomeSize, CircuitGenome::Evaluator);

	// LOAD genome
	string fileName = "EAC_circuit.bin";
	fstream	efile;
	string executetext;
	efile.open(fileName.c_str(), fstream::in);
	
	if (efile.is_open()) {
		getline(efile, executetext);
		CircuitGenome::ExecuteFromText(executetext, &genom);
		efile.close();
	}

    if (status == STAT_OK) {
		ofstream out("EAC_fitnessProgress.txt", ios::app);
		
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
		ga.scoreFilename("scores.log");
		ga.scoreFrequency(1);	// keep the scores of every generation
		ga.flushFrequency(1);	// specify how often to write the score to disk 
		ga.selectScores(GAStatistics::AllScores);

		ga.initialize();

		out << "GAOK" << endl;
		int		actGener = 1;
		int		changed = 1;
		int		evaluateNext = 0;
		
		pGACirc->clearFitnessStats();

		fstream fitfile;

		if (argc > 1 && strcmp(argv[1],"-evolutionoff") == 0)
			genomeTemp = genom;
		
        while (actGener < pBasicSettings.gaConfig.nGeners) {
			//FRACTION FILE FOR BOINC
			fitfile.open("fraction_done.txt", fstream::out | ios::trunc);
			fitfile << ((float)(actGener))/((float)(pBasicSettings.gaConfig.nGeners));
			fitfile.close();
            
			// DO NOT EVOLVE..
			if (argc > 1 && strcmp(argv[1],"-evolutionoff") == 0) {
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
						ofstream ssfile(seedFile.c_str(), ios::app);

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
