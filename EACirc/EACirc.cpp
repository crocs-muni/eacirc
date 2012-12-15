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
    m_status = LoadConfigScript(filename, &basicSettings);
    if (m_status != STAT_OK) {
        mainLogger.out() << "Could not read configuration data from " << FILE_CONFIG << endl;
    }
    // CREATE STRUCTURE OF CIRCUIT FROM BASIC SETTINGS
    pGACirc = &(basicSettings.gaCircuitConfig);
    pGACirc->allocate();

    if (basicSettings.gaCircuitConfig.changeGalibSeedFrequency != 0 &&
            basicSettings.gaCircuitConfig.changeGalibSeedFrequency % basicSettings.gaCircuitConfig.testVectorChangeGener != 0) {
        mainLogger.out() << "GAlib reseeding frequency must be multiple of test vector change frequency." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (basicSettings.gaCircuitConfig.TVCGProgressive == 1 &&
            basicSettings.gaCircuitConfig.changeGalibSeedFrequency != 0) {
        mainLogger.out() << "Prograsive teste vector generation cannot be used when saving state." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }

    if (m_status == STAT_OK) {
        m_readyToRun |= EACIRC_CONFIG_LOADED;
    }
}


void EACirc::saveState(string filename) {
    TiXmlElement* pRoot = new TiXmlElement("eacirc_state");
    TiXmlElement* pElem;

    pElem = new TiXmlElement("generations_required");
    pElem->LinkEndChild(new TiXmlText(to_string(basicSettings.gaConfig.nGeners).c_str()));
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

    m_status = saveXMLFile(pRoot,filename);
    if (m_status != STAT_OK) {
        mainLogger.out() << "Error saving state to file " << filename << "." << endl;
        return;
    }
    mainLogger.out() << "State successfully saved to file " << filename << "." << endl;
}

void EACirc::loadState(string filename) {
    TiXmlNode* pRoot = NULL;
    m_status = loadXMLFile(pRoot,filename);
    if (m_status != STAT_OK) {
        mainLogger.out() << "error: Could not load state from file " << filename << "." << endl;
        return;
    }

    TiXmlElement* pElem = pRoot->FirstChildElement();
    for( pElem; pElem; pElem = pElem->NextSiblingElement()) {
        // restore main seed
        if (string(pElem->Value()) == "main_seed") {
            istringstream(pElem->GetText()) >> m_originalSeed;
            basicSettings.rndGen.randomSeed = m_originalSeed;
        }
        // restore current galib seed
        if (string(pElem->Value()) == "current_galib_seed") {
            istringstream(pElem->GetText()) >> m_currentGalibSeed;
        }
        // initialize random generators (main, quantum, bias)
        if (string(pElem->Value()) == "random_generators") {
            TiXmlElement* pElem2 = pElem->FirstChildElement();
            mainGenerator = IRndGen::parseGenerator(pElem2);
            pElem2 = pElem2->NextSiblingElement();
            rndGen = IRndGen::parseGenerator(pElem2);
            pElem2 = pElem2->NextSiblingElement();
            biasRndGen = IRndGen::parseGenerator(pElem2);
        }
    }
    mainLogger.out() << "State successfully loaded from file " << filename << "." << endl;
}

void EACirc::createState() {
    // INIT MAIN GENERATOR
    // if useFixedSeed and proper seed was specified, use it
    if (basicSettings.rndGen.useFixedSeed && basicSettings.rndGen.randomSeed != 0) {
        m_originalSeed = basicSettings.rndGen.randomSeed;
        mainGenerator = new MD5RndGen(m_originalSeed);
        mainLogger.out() << "Using fixed seed: " << m_originalSeed << endl;
    } else {
        // generate random seed, if none provided
        mainGenerator = new MD5RndGen(clock() + time(NULL) + getpid());
        mainGenerator->getRandomFromInterval(ULONG_MAX,&m_originalSeed);
        delete mainGenerator;
        mainGenerator = NULL; // necessary !!! (see guts of MD5RndGen)
        mainGenerator = new MD5RndGen(m_originalSeed);
        mainLogger.out() << "Using system-generated random seed: " << m_originalSeed << endl;
    }

    // INIT RNG
    unsigned long generatorSeed;
    mainGenerator->getRandomFromInterval(ULONG_MAX,&generatorSeed);
    rndGen = new QuantumRndGen(generatorSeed, basicSettings.rndGen.QRBGSPath);
    mainLogger.out() << "Random generator initialized (" << rndGen->shortDescription() << ")" <<endl;
    // INIT BIAS RNDGEN
    mainGenerator->getRandomFromInterval(ULONG_MAX,&generatorSeed);
    biasRndGen = new BiasRndGen(generatorSeed, basicSettings.rndGen.QRBGSPath, basicSettings.rndGen.biasFactor);
    mainLogger.out() << "Bias random generator initialized (" << biasRndGen->shortDescription() << ")" <<endl;

    // GENERATE SEED FOR GALIB
    mainGenerator->getRandomFromInterval(ULONG_MAX,&m_currentGalibSeed);
    mainLogger.out() << "State successfully initialized." << endl;
}

void EACirc::savePopulation(string filename) {
    ofstream populationFile(filename);
    if (!populationFile.is_open()) {
        mainLogger.out() << "Error saving population to file " << filename << "." << endl;
        m_status = STAT_FILE_OPEN_FAIL;
        return;
    }
    populationFile << basicSettings.gaConfig.popSize << endl;
    populationFile << basicSettings.gaCircuitConfig.genomeSize << endl;
    m_status = CircuitGenome::writePopulation(m_gaData->population(),populationFile);
    if (m_status != STAT_OK) {
        mainLogger.out() << "Error saving population to file " << filename << "." << endl;
        populationFile.close();
        return;
    }
    populationFile.close();
    mainLogger.out() << "Population successfully saved to file " << filename << "." << endl;
}

void EACirc::loadPopulation(string filename) {
    GAPopulation population;
    ifstream populationFile(filename);
    if (!populationFile.is_open()) {
        mainLogger.out() << "error: Cannot read population from file " << filename << "." << endl;
        m_status = STAT_FILE_OPEN_FAIL;
    }
    CircuitGenome::readPopulation(population,populationFile);
    seedAndResetGAlib(population);
    mainLogger.out() << "Population successfully loaded from file " << filename << "." << endl;
}

void EACirc::createPopulation() {
    // seed GAlib (initializations may require random numbers)
    GARandomSeed(m_currentGalibSeed);
    // temporary structure for genome (empty or loaded from file)
    GA1DArrayGenome<unsigned long> genome(pGACirc->genomeSize, CircuitGenome::Evaluator);
    genome.initializer(CircuitGenome::Initializer);
    genome.mutator(CircuitGenome::Mutator);
    genome.crossover(CircuitGenome::Crossover);

    // load genome, if needed
    if (m_loadGenome) {
        fstream	genomeFile;
        string executetext;
        genomeFile.open(FILE_GENOME, fstream::in);
        if (!genomeFile.is_open()) {
            m_status = STAT_FILE_OPEN_FAIL;
            mainLogger.out() << "Error: Could not open genome file " << FILE_GENOME << "." << endl;
            return;
        }
        getline(genomeFile, executetext);
        CircuitGenome::ExecuteFromText(executetext, &genome);
        genomeFile.close();
        mainLogger.out() << "Genome successfully loaded file" << FILE_GENOME << "." << endl;

        // create population
        GAPopulation population(genome,1);
        // generate new seed because initializations may have used some randomness
        mainGenerator->getRandomFromInterval(ULONG_MAX,&m_currentGalibSeed);
        seedAndResetGAlib(population);
    } else {
        GAPopulation population(genome,1);
        // create genetic algorithm and initialize population
        seedAndResetGAlib(population);
        m_gaData->initialize();

        // generate new seed because initializations may have used some randomness
        mainGenerator->getRandomFromInterval(ULONG_MAX,&m_currentGalibSeed);
        seedAndResetGAlib(m_gaData->population());
    }
    mainLogger.out() << "Population successfully initialized." << endl;
}

void EACirc::saveProgress(string stateFilename, string populationFilename) {
    if (m_status != STAT_OK) return;
    saveState(stateFilename);
    savePopulation(populationFilename);
}

void EACirc::initializeState() {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & EACIRC_CONFIG_LOADED) != EACIRC_CONFIG_LOADED) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }

    if (basicSettings.loadState) {
        loadState(FILE_STATE);
        loadPopulation(FILE_POPULATION);
    } else {
        createState();
        createPopulation();
    }

    if (m_status == STAT_OK) {
        m_readyToRun |= EACIRC_INITIALIZED;
    }
}

void EACirc::prepare() {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & (EACIRC_CONFIG_LOADED | EACIRC_INITIALIZED)) !=
            (EACIRC_CONFIG_LOADED | EACIRC_INITIALIZED)) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }

    /* temporaroly moving to run cycle, since random keys are generated only on creation of EncryptorDecryptor
      TODO: remake tenerating of test vectors so that IV and KEY is also generated when needed
    mainLogger.out() << "Initializing Evaluator and Encryptor-Decryptor." << endl;
    // INIT EVALUATOR and ENCRYPTOR-DECRYPTOR (according to loaded settings)
    encryptorDecryptor = new EncryptorDecryptor();
    */
    m_evaluator = new Evaluator();

    // PREPARE THE LOGGING FILES
    std::remove(FILE_FITNESS_PROGRESS);
    std::remove(FILE_BEST_FITNESS);
    std::remove(FILE_AVG_FITNESS);

    //LOG THE TESTVECTGENER METHOD
    if (pGACirc->testVectorGenerMethod == ESTREAM_CONST) {
        ofstream out(FILE_FITNESS_PROGRESS, ios::app);
        out << "Using Ecrypt candidate n." << pGACirc->testVectorEstream << " (" <<  basicSettings.gaCircuitConfig.limitAlgRoundsCount << " rounds) AND candidate n." << pGACirc->testVectorEstream2 << " (" << basicSettings.gaCircuitConfig.limitAlgRoundsCount2 << " rounds)" <<  endl;
        out.close();
        mainLogger.out() << "stream1: using " << estreamToString(pGACirc->testVectorEstream);
        mainLogger.out() << " (" << basicSettings.gaCircuitConfig.limitAlgRoundsCount << " rounds)" << endl;
        mainLogger.out() << "stream2: using " << estreamToString(pGACirc->testVectorEstream2);
        mainLogger.out() << " (" << basicSettings.gaCircuitConfig.limitAlgRoundsCount2 << " rounds)" << endl;
    }

    if (m_status == STAT_OK) {
        m_readyToRun |= EACIRC_PREPARED;
    }
}

void EACirc::seedAndResetGAlib(GAPopulation population) {
    // set GAlib seed
    GARandomSeed(m_currentGalibSeed);
    // reset population stats
    population.touch();
    // delete any previous instance of genetic algorithm
    if (m_gaData != NULL) {
        delete m_gaData;
        m_gaData = NULL;
    }
    // create new instance of genetic algorithm
    m_gaData = new GASteadyStateGA(population);
    // initialize the new genetic algorithm
    m_gaData->populationSize(basicSettings.gaConfig.popSize);
    m_gaData->nReplacement(2 * basicSettings.gaConfig.popSize / 3);
    m_gaData->nGenerations(basicSettings.gaConfig.nGeners);
    m_gaData->pCrossover(basicSettings.gaConfig.pCross);
    m_gaData->pMutation(basicSettings.gaConfig.pMutt);
    m_gaData->scoreFilename(FILE_GALIB_SCORES);
    m_gaData->scoreFrequency(1);	// keep the scores of every generation
    m_gaData->flushFrequency(1);	// specify how often to write the score to disk
    m_gaData->selectScores(GAStatistics::AllScores);
    mainLogger.out() << "info: GAlib seeded and reset." << endl;
}

void EACirc::run() {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & (EACIRC_CONFIG_LOADED | EACIRC_INITIALIZED | EACIRC_PREPARED)) !=
            (EACIRC_CONFIG_LOADED | EACIRC_INITIALIZED | EACIRC_PREPARED)) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }

    // SAVE INITIAL STATE
    saveProgress(FILE_STATE_INITIAL,FILE_POPULATION_INITIAL);

    //m_actGener = 1;
    int	changed = 1;
    bool evaluateNow = false;
    pGACirc->clearFitnessStats();
    fstream fitfile;

    GA1DArrayGenome<unsigned long> genome(pGACirc->genomeSize, CircuitGenome::Evaluator);
    GA1DArrayGenome<unsigned long> genomeTemp(pGACirc->genomeSize, CircuitGenome::Evaluator);
    genome = m_gaData->population().individual(0);

    if (m_evolutionOff) {
        genomeTemp = genome;
    }

    mainLogger.out() << "Starting evolution." << endl;
    for (m_actGener = 1; m_actGener <= basicSettings.gaConfig.nGeners; m_actGener++) {

        //FRACTION FILE FOR BOINC
        fitfile.open(FILE_BOINC_FRACTION_DONE, fstream::out | ios::trunc);
        fitfile << ((float)(m_actGener))/((float)(basicSettings.gaConfig.nGeners));
        fitfile.close();

        // DO NOT EVOLVE..
        if (m_evolutionOff) {
            m_evaluator->generateTestVectors();
            m_evaluator->evaluateStep(genome, m_actGener);
            continue;
        }

        // GENERATE TEST VECTORS IF NEEDED
        if (pGACirc->TVCGProgressive) {
            // TODO: understand and correct
            if (changed > m_actGener/pGACirc->testVectorChangeGener + 1) {
                m_evaluator->generateTestVectors();
                evaluateNow = true;
                changed = 0;
            }
        } else {
            if (m_actGener %(pGACirc->testVectorChangeGener) == 1) {

                //temporary
                mainLogger.out() << "info: recreating Encryptor-Decryptor." << endl;
                // INIT EVALUATOR and ENCRYPTOR-DECRYPTOR (according to loaded settings)
                if (encryptorDecryptor != NULL) delete encryptorDecryptor;
                encryptorDecryptor = NULL;
                encryptorDecryptor = new EncryptorDecryptor();

                m_evaluator->generateTestVectors();
            }
            if (m_actGener %(pGACirc->testVectorChangeGener) == 0) {
                evaluateNow = true;
            }
        }
        // variable for computing when TVCGProgressive = true
        changed++;

        // RESET EVALUTION FOR ALL GENOMS
        m_gaData->pop->flushEvalution();
        // GA evolution step
        m_gaData->step();

        genomeTemp = (GA1DArrayGenome<unsigned long>&) m_gaData->population().best();// .statistics().bestIndividual();

        if (evaluateNow || pGACirc->evaluateEveryStep) {
            m_evaluator->evaluateStep(genomeTemp, m_actGener);
            evaluateNow = false;
        }

        // if needed, reseed GAlib and save state and population
        if (basicSettings.gaCircuitConfig.changeGalibSeedFrequency != 0
                && m_actGener % basicSettings.gaCircuitConfig.changeGalibSeedFrequency == 0) {
            mainGenerator->getRandomFromInterval(ULONG_MAX,&m_currentGalibSeed);
            seedAndResetGAlib(m_gaData->population());
            saveProgress(FILE_STATE,FILE_POPULATION);
        }
    }

    // commented for testing purposes of saving state
    // GENERATE FRESH NEW SET AND EVALUATE THE RESULT
    //m_evaluator->generateTestVectors();
    //m_evaluator->evaluateStep(genomeTemp, m_actGener);

    //Print the best circuit
    //CircuitGenome::PrintCircuit(genomeTemp,FILE_BEST_CIRCUIT,0,1);
    //saveState(FILE_STATE,FILE_POPULATION);
}
