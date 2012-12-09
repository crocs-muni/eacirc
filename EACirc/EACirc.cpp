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
#include "random_generator/MD5RndGen.h"
//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
#include "XMLProcessor.h"
#include "Evaluator.h"
#include "CircuitGenome.h"
#include "estream/estreamInterface.h"
#include "test_vector_generator/ITestVectGener.h"
#include "test_vector_generator/EstreamTestVectGener.h"
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

IRndGen*                rndGen = NULL;
IRndGen*                biasRndGen = NULL;
GA_CIRCUIT*             pGACirc = NULL;
EncryptorDecryptor*		encryptorDecryptor = NULL;

EACirc::EACirc()
    : m_status(STAT_OK), m_evolutionOff(false), m_loadGenome(false),
      m_seed(0), m_evaluator(NULL), m_readyToRun(0) {
    pGACirc = new GA_CIRCUIT;
}

EACirc::EACirc(bool evolutionOff)
    : m_status(STAT_OK), m_evolutionOff(evolutionOff), m_loadGenome(evolutionOff),
      m_seed(0), m_evaluator(NULL), m_readyToRun(0) {
    // load genome, if evolution is off
}

EACirc::~EACirc() {
    delete m_evaluator;
    delete encryptorDecryptor;
    pGACirc->release();
    delete rndGen;
    delete biasRndGen;
}

int EACirc::getStatus() {
    return m_status;
}

void EACirc::loadConfiguration(string filename) {
    if (m_status != STAT_OK) return;
    m_status = LoadConfigScript(filename, &pBasicSettings);
    if (m_status != STAT_OK) {
        mainLogger.out() << "Could not read configuration data from " << FILE_CONFIG << endl;
    }
    // CREATE STRUCTURE OF CIRCUIT FROM BASIC SETTINGS
    pGACirc = &(pBasicSettings.gaCircuitConfig);
    pGACirc->allocate();
    m_readyToRun |= EACIRC_CONFIG_LOADED;
}

void EACirc::loadState(string filename) {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & EACIRC_CONFIG_LOADED) != EACIRC_CONFIG_LOADED) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }
    // set seed
    // init rng (3x)
    // set to load genome

    mainLogger.out() << "Not implemented yet." << endl;
    m_status = STAT_NOT_IMPLEMENTED_YET;

    // m_readyToRun |= EACIRC_INITIALIZED
}

void EACirc::saveState(string filename) {
    if (m_status != STAT_OK) return;
    TiXmlElement* pRoot = new TiXmlElement("random_generators");
    pRoot->LinkEndChild(mainGenerator->exportGenerator());
    pRoot->LinkEndChild(rndGen->exportGenerator());
    pRoot->LinkEndChild(biasRndGen->exportGenerator());
    saveXMLFile(pRoot,filename);
}

void EACirc::initializeState() {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & EACIRC_CONFIG_LOADED) != EACIRC_CONFIG_LOADED) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }

    /*
    // RESTORE THE SEED
    fstream	sfile;
    string sseed;
    */

    //with useFixedSeed, a seed file is used, upon fail, randomseed argument is used
    if (pBasicSettings.rndGen.useFixedSeed && pBasicSettings.rndGen.randomSeed != 0) {
        /* // previous solution: first try to load LastSeed.txt, then seed from config.xml, then new random
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
        */
        m_seed = pBasicSettings.rndGen.randomSeed;
        mainGenerator = new MD5RndGen(m_seed);
        mainLogger.out() << "Using fixed seed: " << m_seed << endl;
    } else {
        //generate random seed, if none provided
        mainGenerator = new MD5RndGen(clock() + time(NULL) + getpid());
        mainGenerator->getRandomFromInterval(ULONG_MAX,&m_seed);
        mainLogger.out() << "Using system-generated random seed: " << m_seed << endl;
    }

    //INIT RNG
    rndGen = new QuantumRndGen(m_seed, pBasicSettings.rndGen.QRBGSPath);
    mainLogger.out() << "Random generator initialized (" << rndGen->shortDescription() << ")" <<endl;
    //INIT BIAS RNDGEN
    biasRndGen = new BiasRndGen(m_seed, pBasicSettings.rndGen.QRBGSPath, pBasicSettings.rndGen.biasFactor);
    mainLogger.out() << "Bias random generator initialized (" << biasRndGen->shortDescription() << ")" <<endl;

    m_readyToRun |= EACIRC_INITIALIZED;
}

void EACirc::prepare() {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & (EACIRC_CONFIG_LOADED | EACIRC_INITIALIZED)) !=
            (EACIRC_CONFIG_LOADED | EACIRC_INITIALIZED)) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }

    mainLogger.out() << "Initializing Evaluator and Encryptor-Decryptor." << endl;
    // INIT EVALUATOR and ENCRYPTOR-DECRYPTOR (according to loaded settings)
    encryptorDecryptor = new EncryptorDecryptor();
    m_evaluator = new Evaluator();

    // PREPARE THE LOGGING FILES
    std::remove(FILE_FITNESS_PROGRESS);
    std::remove(FILE_BEST_FITNESS);
    std::remove(FILE_AVG_FITNESS);

    /*
    //LOG THE SEED
    ofstream ssfile(FILE_SEEDFILE, ios::app);
    ssfile << "----------";
    if (pBasicSettings.rndGen.useFixedSeed)
        ssfile << "Using fixed seed" << endl;
    else
        ssfile << "Using random seed" << endl;
    ssfile << m_seed << endl;
    ssfile.close();
    */

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

    m_readyToRun |= EACIRC_PREPARED;
}

void EACirc::run() {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & (EACIRC_CONFIG_LOADED | EACIRC_INITIALIZED | EACIRC_PREPARED)) !=
            (EACIRC_CONFIG_LOADED | EACIRC_INITIALIZED | EACIRC_PREPARED)) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }

    // SAVE INITIAL STATE
    saveState(FILE_STATE_INITIAL);

    // ***** GAlib INITIALIZATION *****
    mainLogger.out() << "Initialising GAlib." << endl;
    GARandomSeed(m_seed);
    // CREATE GA STRUCTS
    GA1DArrayGenome<unsigned long> genom(pGACirc->genomeSize, CircuitGenome::Evaluator);
    GA1DArrayGenome<unsigned long> genomeTemp(pGACirc->genomeSize, CircuitGenome::Evaluator);
    // INIT GENOME STRUCTURES
    genom.initializer(CircuitGenome::Initializer);
    genom.mutator(CircuitGenome::Mutator);
    genom.crossover(CircuitGenome::Crossover);
    // LOAD genome
    if (m_loadGenome) {
        fstream	genomeFile;
        string executetext;
        genomeFile.open(FILE_GENOME, fstream::in);
        if (genomeFile.is_open()) {
            mainLogger.out() << "Loading genome from file." << endl;
            getline(genomeFile, executetext);
            CircuitGenome::ExecuteFromText(executetext, &genom);
            genomeFile.close();
        } else {
            m_status = STAT_FILE_OPEN_FAIL;
            return;
        }
    }
    // INIT MAIN GA
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
    //out << "GAOK" << endl;
    mainLogger.out() << "GAlib fully initialized." << endl;
    // ***** END GAlib INITIALIZATIONS *****

    int actGener = 1;
    int	changed = 1;
    int	evaluateNext = 0;
    pGACirc->clearFitnessStats();
    fstream fitfile;

    if (m_evolutionOff) {
        genomeTemp = genom;
    }

    mainLogger.out() << "Starting evolution." << endl;
    while (actGener < pBasicSettings.gaConfig.nGeners) {
        actGener++;

        //FRACTION FILE FOR BOINC
        fitfile.open(FILE_BOINC_FRACTION_DONE, fstream::out | ios::trunc);
        fitfile << ((float)(actGener))/((float)(pBasicSettings.gaConfig.nGeners));
        fitfile.close();

        // WHY creating new evaluator?
        // and why new for every generation?
        //Evaluator *evaluator = new Evaluator();

        // DO NOT EVOLVE..
        if (m_evolutionOff) {
            m_evaluator->generateTestVectors();
            m_evaluator->evaluateStep(genom, actGener);
        } else {
            // RESET EVALUTION FOR ALL GENOMS
            ga.pop->flushEvalution();
            ga.step(); // GA evolution step

            if (evaluateNext) {
                m_evaluator->evaluateStep(genomeTemp, actGener);
                evaluateNext = 0;
            }

            genomeTemp = (GA1DArrayGenome<unsigned long>&) ga.population().best();// .statistics().bestIndividual();
            
            if ((pGACirc->TVCGProgressive && (changed > actGener/pGACirc->testVectorChangeGener + 1)) ||
                    (!pGACirc->TVCGProgressive && ((actGener %(pGACirc->testVectorChangeGener)) == 0))) {

                if (pGACirc->changeGalibSeedFrequency != 0 &&
                        actGener % pGACirc->changeGalibSeedFrequency == 0) {
                    saveState(FILE_STATE);
                    rndGen->getRandomFromInterval(ULONG_MAX, &m_seed);
                    GARandomSeed(m_seed);
                    mainLogger.out() << "GAlib reseeded (actGener = " << actGener << ")" << endl;
                }

                // GENERATE FRESH SET AND EVALUATE ONLY THE BEST ONE
                m_evaluator->generateTestVectors();
                evaluateNext = 1;
                changed = 0;
            }
            else if (pGACirc->evaluateEveryStep) m_evaluator->evaluateStep(genomeTemp, actGener);
            changed++;
        }
    }

    // GENERATE FRESH NEW SET AND EVALUATE THE RESULT
    m_evaluator->generateTestVectors();
    m_evaluator->evaluateStep(genomeTemp, actGener);

    //Print the best circuit
    CircuitGenome::PrintCircuit(genomeTemp,FILE_BEST_CIRCUIT,0,1);
}
