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
#include "random_generator/QuantumRndGen.h"
//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
#include "XMLProcessor.h"
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
                mainLogger.out() << "  " << CMD_OPT_LOGGING << "  (set logging to clog)" << endl;
                mainLogger.out() << "  " << CMD_OPT_LOGGING_TO_FILE << "  (set logging to logfile)" << endl;
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
        // cout << sseed << endl;
		if (!sseed.empty())
			seed = atoi(sseed.c_str());
        sfile.close();

		// USE STATIC SEED
        if (!seed) {
            seed = pBasicSettings.rndGen.randomSeed;
            mainLogger.out() << "Using fixed seed: " << seed << endl;
        }
	}

    IRndGen::initMainGenerator(clock() + time(NULL) + getpid());
    if (seed == 0){
        seed = (IRndGen::getRandomFromMainGenerator() %100000) + ((IRndGen::getRandomFromMainGenerator() %42946) *100000);
        mainLogger.out() << "Using system-generated random seed: " << seed << endl;
    }
	//INIT RNG
	GARandomSeed(seed);
    rndGen = new QuantumRndGen(seed, pBasicSettings.rndGen.QRBGSPath);
    mainLogger.out() << "Random generator initialized (" << rndGen->shortDescription() << ")" <<endl;
    //INIT BIAS RNDGEN
    biasRndGen = new BiasRndGen(seed, pBasicSettings.rndGen.QRBGSPath, pBasicSettings.rndGen.biasFactor);
    mainLogger.out() << "Bias random generator initialized (" << biasRndGen->shortDescription() << ")" <<endl;

    //mainLogger.out() << "rndgen: " << *rndGen << endl;
    saveXMLFile(biasRndGen->exportGenerator(),"generator.xml");

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

                        rndGen->getRandomFromInterval(4294967295, &seed);
						GARandomSeed(seed);
                        // WHY RESEEDING RANDOM GENERATOR? (and why not reseeding bias generator as well?)
                        //orig: rndGen->InitRandomGenerator(seed,pBasicSettings.rndGen.QRBGSPath);
                        delete rndGen;
                        rndGen = new QuantumRndGen(seed,pBasicSettings.rndGen.QRBGSPath);
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
