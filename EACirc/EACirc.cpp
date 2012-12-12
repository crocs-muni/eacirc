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
      m_originalSeed(0), m_currentGalibSeed(0), m_evaluator(NULL), m_gaData(NULL),
      m_readyToRun(0), m_actGener(0) {
    pGACirc = new GA_CIRCUIT;
}

EACirc::EACirc(bool evolutionOff)
    : m_status(STAT_OK), m_evolutionOff(evolutionOff), m_loadGenome(evolutionOff),
      m_originalSeed(0), m_currentGalibSeed(0), m_evaluator(NULL), m_gaData(NULL),
      m_readyToRun(0), m_actGener(0) {
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

    if (pBasicSettings.gaCircuitConfig.changeGalibSeedFrequency != 0 &&
            pBasicSettings.gaCircuitConfig.changeGalibSeedFrequency % pBasicSettings.gaCircuitConfig.testVectorChangeGener != 0) {
        mainLogger.out() << "GAlib reseeding frequency must be multiple of test vector change frequency." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }

    if (m_status == STAT_OK) {
        m_readyToRun |= EACIRC_CONFIG_LOADED;
    }
}

void EACirc::loadState(string stateFilename, string populationFilename) {
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

void EACirc::saveState(string stateFilename, string populationFilename) {
    if (m_status != STAT_OK) return;

    // SAVING STATE

    TiXmlElement* pRoot = new TiXmlElement("eacirc_state");
    TiXmlElement* pElem;

    pElem = new TiXmlElement("generations_required");
    pElem->LinkEndChild(new TiXmlText(to_string(pBasicSettings.gaConfig.nGeners).c_str()));
    pRoot->LinkEndChild(pElem);
    pElem = new TiXmlElement("generations_finished");
    pElem->LinkEndChild(new TiXmlText(to_string(m_actGener).c_str()));
    pRoot->LinkEndChild(pElem);
    pElem = new TiXmlElement("main_seed");
    pElem->LinkEndChild(new TiXmlText(to_string(m_originalSeed).c_str()));
    pRoot->LinkEndChild(pElem);
    pElem = new TiXmlElement("current_galib_seed");
    pElem->LinkEndChild(new TiXmlText(to_string(m_currentGalibSeed).c_str()));
    pRoot->LinkEndChild(pElem);

    pElem = new TiXmlElement("random_generators");
    pElem->LinkEndChild(mainGenerator->exportGenerator());
    pElem->LinkEndChild(rndGen->exportGenerator());
    pElem->LinkEndChild(biasRndGen->exportGenerator());
    pRoot->LinkEndChild(pElem);

    int status = saveXMLFile(pRoot,stateFilename);
    if (status != STAT_OK) {
        mainLogger.out() << "Error saving state to file " << stateFilename << "." << endl;
        m_status = status;
    }

    // SAVING POPULATION
    ofstream populationFile(populationFilename);
    if (populationFile.is_open()) {
        populationFile << pBasicSettings.gaConfig.nGeners << endl;
        populationFile << pBasicSettings.gaCircuitConfig.genomeSize << endl;
        CircuitGenome::writePopulation(m_gaData->population(),populationFile);
        populationFile.close();
    } else {
        mainLogger.out() << "Error saving population to file " << populationFilename << "." << endl;
        m_status = STAT_FILE_OPEN_FAIL;
    }
}

void EACirc::initializeState() {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & EACIRC_CONFIG_LOADED) != EACIRC_CONFIG_LOADED) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }

    //with useFixedSeed, a seed file is used, upon fail, randomseed argument is used
    if (pBasicSettings.rndGen.useFixedSeed && pBasicSettings.rndGen.randomSeed != 0) {
        m_originalSeed = pBasicSettings.rndGen.randomSeed;
        mainGenerator = new MD5RndGen(m_originalSeed);
        mainLogger.out() << "Using fixed seed: " << m_originalSeed << endl;
    } else {
        //generate random seed, if none provided
        mainGenerator = new MD5RndGen(clock() + time(NULL) + getpid());
        mainGenerator->getRandomFromInterval(ULONG_MAX,&m_originalSeed);
        delete mainGenerator;
        mainGenerator = NULL; // necessary !!!
        mainGenerator = new MD5RndGen(m_originalSeed);
        mainLogger.out() << "Using system-generated random seed: " << m_originalSeed << endl;
    }

    //INIT RNG
    unsigned long generatorSeed;
    mainGenerator->getRandomFromInterval(ULONG_MAX,&generatorSeed);
    rndGen = new QuantumRndGen(generatorSeed, pBasicSettings.rndGen.QRBGSPath);
    mainLogger.out() << "Random generator initialized (" << rndGen->shortDescription() << ")" <<endl;
    //INIT BIAS RNDGEN
    mainGenerator->getRandomFromInterval(ULONG_MAX,&generatorSeed);
    biasRndGen = new BiasRndGen(generatorSeed, pBasicSettings.rndGen.QRBGSPath, pBasicSettings.rndGen.biasFactor);
    mainLogger.out() << "Bias random generator initialized (" << biasRndGen->shortDescription() << ")" <<endl;

    // ***** GAlib INITIALIZATION *****
    mainLogger.out() << "Initialising GAlib." << endl;
    mainGenerator->getRandomFromInterval(ULONG_MAX,&m_currentGalibSeed);
    GARandomSeed(m_currentGalibSeed);
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
    m_gaData = new GASteadyStateGA(genom);
    m_gaData->populationSize(pBasicSettings.gaConfig.popSize);
    m_gaData->nReplacement(2 * pBasicSettings.gaConfig.popSize / 3);
    m_gaData->nGenerations(pBasicSettings.gaConfig.nGeners);
    m_gaData->pCrossover(pBasicSettings.gaConfig.pCross);
    m_gaData->pMutation(pBasicSettings.gaConfig.pMutt);
    m_gaData->scoreFilename(FILE_GALIB_SCORES);
    m_gaData->scoreFrequency(1);	// keep the scores of every generation
    m_gaData->flushFrequency(1);	// specify how often to write the score to disk
    m_gaData->selectScores(GAStatistics::AllScores);
    m_gaData->initialize();
    //out << "GAOK" << endl;
    mainLogger.out() << "GAlib fully initialized." << endl;
    // ***** END GAlib INITIALIZATIONS *****

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
    saveState(FILE_STATE_INITIAL,FILE_POPULATION_INITIAL);

    m_actGener = 0;
    int	changed = 1;
    int	evaluateNext = 0;
    pGACirc->clearFitnessStats();
    fstream fitfile;

    GA1DArrayGenome<unsigned long> genome(pGACirc->genomeSize, CircuitGenome::Evaluator);
    GA1DArrayGenome<unsigned long> genomeTemp(pGACirc->genomeSize, CircuitGenome::Evaluator);
    genome = m_gaData->population().individual(0);

    if (m_evolutionOff) {
        genomeTemp = genome;
    }

    mainLogger.out() << "Starting evolution." << endl;
    while (m_actGener < pBasicSettings.gaConfig.nGeners) {
        m_actGener++;

        //FRACTION FILE FOR BOINC
        fitfile.open(FILE_BOINC_FRACTION_DONE, fstream::out | ios::trunc);
        fitfile << ((float)(m_actGener))/((float)(pBasicSettings.gaConfig.nGeners));
        fitfile.close();

        // WHY creating new evaluator?
        // and why new for every generation?
        //Evaluator *evaluator = new Evaluator();

        // DO NOT EVOLVE..
        if (m_evolutionOff) {
            m_evaluator->generateTestVectors();
            m_evaluator->evaluateStep(genome, m_actGener);
        } else {
            // RESET EVALUTION FOR ALL GENOMS
            m_gaData->pop->flushEvalution();
            m_gaData->step(); // GA evolution step

            if (evaluateNext) {
                m_evaluator->evaluateStep(genomeTemp, m_actGener);
                evaluateNext = 0;
            }

            genomeTemp = (GA1DArrayGenome<unsigned long>&) m_gaData->population().best();// .statistics().bestIndividual();
            
            if ((pGACirc->TVCGProgressive && (changed > m_actGener/pGACirc->testVectorChangeGener + 1)) ||
                    (!pGACirc->TVCGProgressive && ((m_actGener %(pGACirc->testVectorChangeGener)) == 0))) {

                /* original location
                if (pGACirc->changeGalibSeedFrequency != 0 &&
                        actGener % pGACirc->changeGalibSeedFrequency == 0) {
                    saveState(FILE_STATE);
                    rndGen->getRandomFromInterval(ULONG_MAX, &m_seed);
                    GARandomSeed(m_seed);
                    mainLogger.out() << "GAlib reseeded (actGener = " << actGener << ")" << endl;
                }
                */

                // GENERATE FRESH SET AND EVALUATE ONLY THE BEST ONE
                m_evaluator->generateTestVectors();
                evaluateNext = 1;
                changed = 0;
            }
            else if (pGACirc->evaluateEveryStep) m_evaluator->evaluateStep(genomeTemp, m_actGener);
            changed++;
        }

        if (pBasicSettings.gaCircuitConfig.changeGalibSeedFrequency != 0
                && m_actGener % pBasicSettings.gaCircuitConfig.changeGalibSeedFrequency == 0) {
            mainGenerator->getRandomFromInterval(ULONG_MAX, &m_currentGalibSeed);
            GARandomSeed(m_currentGalibSeed);
            mainLogger.out() << "info: GAlib reseeded (actGener = " << m_actGener << ")" << endl;
            saveState(FILE_STATE,FILE_POPULATION);
        }
    }

    // GENERATE FRESH NEW SET AND EVALUATE THE RESULT
    m_evaluator->generateTestVectors();
    m_evaluator->evaluateStep(genomeTemp, m_actGener);

    //Print the best circuit
    CircuitGenome::PrintCircuit(genomeTemp,FILE_BEST_CIRCUIT,0,1);
    saveState(FILE_STATE,FILE_POPULATION);
}
