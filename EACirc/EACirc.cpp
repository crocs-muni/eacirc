#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>

#include "EACirc.h"
#include "EACglobals.h"
#include "CommonFnc.h"
#include "generators/IRndGen.h"
#include "generators/BiasRndGen.h"
#include "generators/QuantumRndGen.h"
#include "generators/MD5RndGen.h"
//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
#include "XMLProcessor.h"
#include "CircuitGenome.h"
#include "projects/IProject.h"
#include "evaluators/IEvaluator.h"

#ifdef _WIN32
	#include <Windows.h>
	#define getpid() GetCurrentProcessId()
#endif
#ifdef __linux__
	#include <sys/types.h>
	#include <unistd.h>
#endif

EACirc::EACirc()
    : m_status(STAT_OK), m_originalSeed(0), m_currentGalibSeed(0), m_project(NULL), m_gaData(NULL),
      m_readyToRun(0), m_actGener(0), m_oldGenerations(0) {
    mainLogger.out(LOGGER_INFO) << "EACirc framework started (build unknown)." << endl;
    if (pGlobals != NULL) {
        mainLogger.out(LOGGER_WARNING) << "Globals not NULL. Overwriting." << endl;
    }
    pGlobals = new GLOBALS;
}

EACirc::~EACirc() {
    if (m_gaData) delete m_gaData;
    m_gaData = NULL;
    if (m_project) delete m_project;
    m_project = NULL;
    if (pGlobals->evaluator != NULL &&  m_settings.main.evaluatorType < EVALUATOR_PROJECT_SPECIFIC_MINIMUM) {
        delete pGlobals->evaluator;
        pGlobals->evaluator = NULL;
    }
    if (pGlobals) {
        pGlobals->testVectors.release();
        delete pGlobals;
    }
    pGlobals = NULL;
    if (rndGen) delete rndGen;
    rndGen = NULL;
    if (biasRndGen) delete biasRndGen;
    biasRndGen = NULL;
    if (galibGenerator) delete galibGenerator;
    galibGenerator = NULL;
    if (mainGenerator) delete mainGenerator;
    mainGenerator = NULL;
}

int EACirc::getStatus() const {
    return m_status;
}

void EACirc::loadConfiguration(const string filename) {
    if (m_status != STAT_OK) return;

    // load file
    TiXmlNode* pRoot = NULL;
    m_status = loadXMLFile(pRoot, filename);
    if (m_status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "cannot load configuration." << endl;
        return;
    }

    LoadConfigScript(pRoot, &m_settings);
    if (m_status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Could not read configuration data from file (" << filename << ")." << endl;
    }
    // CREATE STRUCTURE OF CIRCUIT FROM BASIC SETTINGS
    pGlobals->settings = &m_settings;

    // CHECK SETTINGS CONSISTENCY
    checkConfigurationConsistency();
    if (m_status != STAT_OK) return;
    mainLogger.out(LOGGER_INFO) << "Configuration successfully loaded (" << filename << ")." << endl;

    m_project = IProject::getProject(m_settings.main.projectType);
    if (m_project == NULL) {
        m_status = STAT_PROJECT_ERROR;
        mainLogger.out(LOGGER_ERROR) << "Could not load project." << endl;
        return;
    }
    m_status = m_project->loadProjectConfiguration(pRoot);
    mainLogger.out(LOGGER_INFO) << "Project configuration loaded. (" << m_project->shortDescription() << ")" << endl;

    // allocate space for testVecotrs
    pGlobals->testVectors.allocate();

    // write configuration to file with standard name (compatibility of results file), free pRoot
    if (filename != FILE_CONFIG) {
        m_status = saveXMLFile(pRoot,FILE_CONFIG);
    } else {
        delete pRoot;
    }

    if (m_status == STAT_OK) {
        m_readyToRun |= EACIRC_CONFIG_LOADED;
    }
}

void EACirc::checkConfigurationConsistency() {
    if (m_settings.testVectors.outputLength != m_settings.circuit.sizeOutputLayer) {
        mainLogger.out(LOGGER_WARNING) << "Circuit output size does not equal test vector output size." << endl;
    }
    if (m_settings.circuit.sizeLayer > MAX_LAYER_SIZE || m_settings.circuit.numConnectors > MAX_LAYER_SIZE) {
        mainLogger.out(LOGGER_ERROR) << "Maximum layer size exceeded (internal layer size or connector number)." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (m_settings.circuit.sizeLayer < m_settings.circuit.sizeOutputLayer) {
        mainLogger.out(LOGGER_ERROR) << "Circuit output layer size is less than internal layer size." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (m_settings.testVectors.inputLength < m_settings.circuit.sizeInputLayer) {
        mainLogger.out(LOGGER_ERROR) << "Test vector input length is smaller than circuit input layer." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
/* Invalid check, if circuit with memory is used, total number of outputs is computed as m_settings.circuit.memorySize +  m_settings.circuit.totalSizeOutputLayer
    if (m_settings.circuit.useMemory && (m_settings.circuit.memorySize > m_settings.circuit.totalSizeOutputLayer)) {
        mainLogger.out(LOGGER_ERROR) << "Circuit memory too large, larger than circuit output." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
	*/
    if (m_settings.circuit.useMemory && (m_settings.circuit.memorySize >= m_settings.circuit.sizeInputLayer)) {
        mainLogger.out(LOGGER_ERROR) << "Circuit memory too large, larger than circuit input (or equal)." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (m_settings.main.recommenceComputation && !m_settings.main.loadInitialPopulation) {
        mainLogger.out(LOGGER_ERROR) << "Initial population must be loaded from file when recommencing computation." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (m_settings.ga.evolutionOff && !m_settings.main.loadInitialPopulation) {
        mainLogger.out(LOGGER_ERROR) << "Initial population must be loaded from file when evolution is off." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (m_settings.main.saveStateFrequency != 0 &&
            m_settings.main.saveStateFrequency % m_settings.testVectors.setChangeFrequency != 0) {
        mainLogger.out(LOGGER_ERROR) << "GAlib reseeding frequency must be multiple of test vector change frequency." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (m_settings.testVectors.setChangeProgressive &&
            m_settings.main.saveStateFrequency != 0) {
        mainLogger.out(LOGGER_ERROR) << "Progressive test vector generation cannot be used when saving state." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
}

void EACirc::saveState(const string filename) {
    TiXmlElement* pRoot = new TiXmlElement("eacirc_state");
    TiXmlElement* pElem;
    TiXmlElement* pElem2;

    pElem = new TiXmlElement("generations_required");
    pElem->LinkEndChild(new TiXmlText(toString(m_settings.main.numGenerations + m_oldGenerations).c_str()));
    pRoot->LinkEndChild(pElem);
    pElem = new TiXmlElement("generations_finished");
    pElem->LinkEndChild(new TiXmlText(toString(m_actGener + m_oldGenerations).c_str()));
    pRoot->LinkEndChild(pElem);
    pElem = new TiXmlElement("main_seed");
    pElem->LinkEndChild(new TiXmlText(toString(m_originalSeed).c_str()));
    pRoot->LinkEndChild(pElem);
    pElem = new TiXmlElement("current_galib_seed");
    pElem->LinkEndChild(new TiXmlText(toString(m_currentGalibSeed).c_str()));
    pRoot->LinkEndChild(pElem);

    pElem = new TiXmlElement("random_generators");
    pElem2 = new TiXmlElement("main_generator");
    pElem2->LinkEndChild(mainGenerator->exportGenerator());
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("galib_generator");
    pElem2->LinkEndChild(galibGenerator->exportGenerator());
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("rndgen");
    pElem2->LinkEndChild(rndGen->exportGenerator());
    pElem->LinkEndChild(pElem2);
    pElem2 = new TiXmlElement("biasrndgen");
    pElem2->LinkEndChild(biasRndGen->exportGenerator());
    pElem->LinkEndChild(pElem2);
    pRoot->LinkEndChild(pElem);

    // save project
    pRoot->LinkEndChild(m_project->saveProjectStateMain());

    m_status = saveXMLFile(pRoot,filename);
    if (m_status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Cannot save state to file " << filename << "." << endl;
    } else {
        mainLogger.out(LOGGER_INFO) << "State successfully saved to file " << filename << "." << endl;
    }
}

void EACirc::loadState(const string filename) {
    TiXmlNode* pRoot = NULL;
    m_status = loadXMLFile(pRoot,filename);
    if (m_status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Could not load state from file " << filename << "." << endl;
        return;
    }

    // restore generations done
    m_oldGenerations = atol(getXMLElementValue(pRoot,"generations_finished").c_str());
    // restore main seed
    istringstream(getXMLElementValue(pRoot,"main_seed")) >> m_originalSeed;
    m_settings.random.seed = m_originalSeed;
    // restore current galib seed
    istringstream(getXMLElementValue(pRoot,"current_galib_seed")) >> m_currentGalibSeed;
    // initialize random generators (main, quantum, bias)
    mainGenerator = IRndGen::parseGenerator(getXMLElement(pRoot,"random_generators/main_generator/generator"));
    galibGenerator = IRndGen::parseGenerator(getXMLElement(pRoot,"random_generators/galib_generator/generator"));
    rndGen = IRndGen::parseGenerator(getXMLElement(pRoot,"random_generators/rndgen/generator"));
    biasRndGen = IRndGen::parseGenerator(getXMLElement(pRoot,"random_generators/biasrndgen/generator"));

    // load project
    m_status = m_project->loadProjectStateMain(getXMLElement(pRoot,"project"));

    delete pRoot;
    mainLogger.out(LOGGER_INFO) << "State successfully loaded from file " << filename << "." << endl;
}

void EACirc::createState() {
    // INIT MAIN GENERATOR
    // if useFixedSeed and proper seed was specified, use it
    if (m_settings.random.useFixedSeed && m_settings.random.seed != 0) {
        m_originalSeed = m_settings.random.seed;
        mainGenerator = new MD5RndGen(m_originalSeed);
        mainLogger.out(LOGGER_INFO) << "Using fixed seed: " << m_originalSeed << endl;
    } else {
        // generate random seed, if none provided
        mainGenerator = new MD5RndGen(clock() + time(NULL) + getpid());
        mainGenerator->getRandomFromInterval(ULONG_MAX,&m_originalSeed);
        delete mainGenerator;
        mainGenerator = NULL; // necessary !!! (see guts of MD5RndGen)
        mainGenerator = new MD5RndGen(m_originalSeed);
        mainLogger.out(LOGGER_INFO) << "Using system-generated random seed: " << m_originalSeed << endl;
    }

    unsigned long generatorSeed;
    // INIT GALIBRNG
    mainGenerator->getRandomFromInterval(ULONG_MAX,&generatorSeed);
    galibGenerator = new QuantumRndGen(generatorSeed, m_settings.random.qrngPath);
    mainLogger.out(LOGGER_INFO) << "GAlib generator initialized (" << galibGenerator->shortDescription() << ")" << endl;
    // INIT RNG
    mainGenerator->getRandomFromInterval(ULONG_MAX,&generatorSeed);
    rndGen = new QuantumRndGen(generatorSeed, m_settings.random.qrngPath);
    mainLogger.out(LOGGER_INFO) << "Random generator initialized (" << rndGen->shortDescription() << ")" << endl;
    // INIT BIAS RNDGEN
    mainGenerator->getRandomFromInterval(ULONG_MAX,&generatorSeed);
    biasRndGen = new BiasRndGen(generatorSeed, m_settings.random.qrngPath, m_settings.random.biasRndGenFactor);
    mainLogger.out(LOGGER_INFO) << "Bias random generator initialized (" << biasRndGen->shortDescription() << ")" << endl;

    // GENERATE SEED FOR GALIB
    galibGenerator->getRandomFromInterval(ULONG_MAX,&m_currentGalibSeed);
    mainLogger.out(LOGGER_INFO) << "State successfully initialized." << endl;
    // INIT PROJECT STATE
    m_project->initializeProjectState();
    mainLogger.out(LOGGER_INFO) << "Project intial state setup successful (" << m_project->shortDescription() << ")." << endl;
}

void EACirc::savePopulation(const string filename) {
    TiXmlElement* pRoot = CircuitGenome::populationHeader(m_settings.ga.popupationSize);
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population");
    string textCircuit;
    for (int i = 0; i < m_settings.ga.popupationSize; i++) {
        // note: it is not necessary to call individual i in SCALED order
        //       however then the population files differ in order ('diff' cannot be used to finding bugs)
        GA1DArrayGenome<unsigned long>* pGenome = (GA1DArrayGenome<unsigned long>*) &(m_gaData->population().individual(i,GAPopulation::SCALED));
        m_status = CircuitGenome::writeGenome(*pGenome ,textCircuit);
        if (m_status != STAT_OK) {
            mainLogger.out(LOGGER_ERROR) << "Could not save genome in population to file " << filename << "." << endl;
            return;
        }
        pElem2 = new TiXmlElement("genome");
        pElem2->LinkEndChild(new TiXmlText(textCircuit.c_str()));
        pElem->LinkEndChild(pElem2);
    }
    pRoot->LinkEndChild(pElem);

    m_status = saveXMLFile(pRoot, filename);
    if (m_status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Cannot save population to file " << filename << "." << endl;
    } else {
        mainLogger.out(LOGGER_INFO) << "Population successfully saved to file " << filename << "." << endl;
    }
}

void EACirc::loadPopulation(const string filename) {
    TiXmlNode* pRoot = NULL;
    m_status = loadXMLFile(pRoot,filename);
    if (m_status != STAT_OK) {
        mainLogger.out(LOGGER_ERROR) << "Could not load state from file " << filename << "." << endl;
        return;
    }
    int savedPopulationSize = atoi(getXMLElementValue(pRoot,"population_size").c_str());

    int settingsValue;
    settingsValue = atoi(getXMLElementValue(pRoot,"circuit_dimensions/num_layers").c_str());
    if (m_settings.circuit.numLayers != settingsValue) {
        mainLogger.out(LOGGER_ERROR) << "Cannot load population - incompatible number of layers (";
        mainLogger.out() << m_settings.circuit.numLayers << " vs. " << settingsValue << ")." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    settingsValue = atoi(getXMLElementValue(pRoot,"circuit_dimensions/size_layer").c_str());
    if (m_settings.circuit.sizeLayer != settingsValue) {
        mainLogger.out(LOGGER_ERROR) << "Cannot load population - incompatible layer size (";
        mainLogger.out() << m_settings.circuit.sizeLayer << " vs. " << settingsValue << ")." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    settingsValue = atoi(getXMLElementValue(pRoot,"circuit_dimensions/size_input_layer").c_str());
    if (m_settings.circuit.sizeInputLayer != settingsValue) {
        mainLogger.out(LOGGER_ERROR) << "Cannot load population - incompatible input layer size (";
        mainLogger.out() << m_settings.circuit.sizeInputLayer << " vs. " << settingsValue << ")." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    settingsValue = atoi(getXMLElementValue(pRoot,"circuit_dimensions/size_output_layer").c_str());
    if (m_settings.circuit.sizeOutputLayer != settingsValue) {
        mainLogger.out(LOGGER_ERROR) << "Cannot load population - incompatible output layer size (";
        mainLogger.out() << m_settings.circuit.sizeOutputLayer << " vs. " << settingsValue << ")." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (m_status != STAT_OK) {
        delete pRoot;
        return;
    }
    GAPopulation population;
    GA1DArrayGenome<unsigned long> genome(m_settings.circuit.genomeSize, CircuitGenome::Evaluator);
    // INIT GENOME STRUCTURES
    genome.initializer(CircuitGenome::Initializer);
    genome.mutator(CircuitGenome::Mutator);
    genome.crossover(CircuitGenome::Crossover);
    // LOAD genomes
    TiXmlElement* pGenome = getXMLElement(pRoot,"population/genome")->ToElement();
    string textCircuit;
    for (int i = 0; i < savedPopulationSize; i++) {
        if (pGenome->GetText() == NULL) {
            mainLogger.out(LOGGER_ERROR) << "Too few genomes in population - expecting " << savedPopulationSize << "." << endl;
            m_status = STAT_DATA_CORRUPTED;
            delete pRoot;
            return;
        }
        textCircuit = pGenome->GetText();
        m_status = CircuitGenome::readGenomeFromBinary(textCircuit,&genome);
        if (m_status != STAT_OK) return;
        population.add(genome);
        pGenome = pGenome->NextSiblingElement();
    }
    seedAndResetGAlib(population);
    delete pRoot;
    mainLogger.out(LOGGER_INFO) << "Population successfully loaded from file " << filename << "." << endl;
}

void EACirc::createPopulation() {
    if (m_status != STAT_OK) return;
    // seed GAlib (initializations may require random numbers)
    GAResetRNG(m_currentGalibSeed);
    // temporary structure for genome (empty or loaded from file)
    GA1DArrayGenome<unsigned long> genome(m_settings.circuit.genomeSize, CircuitGenome::Evaluator);
    genome.initializer(CircuitGenome::Initializer);
    genome.mutator(CircuitGenome::Mutator);
    genome.crossover(CircuitGenome::Crossover);
    GAPopulation population(genome,m_settings.ga.popupationSize);
    // create genetic algorithm and initialize population
    seedAndResetGAlib(population);
    mainLogger.out(LOGGER_INFO) << "Initializing population." << endl;
    m_gaData->initialize();
    // reset GAlib seed
    galibGenerator->getRandomFromInterval(ULONG_MAX,&m_currentGalibSeed);
    seedAndResetGAlib(m_gaData->population());

    mainLogger.out(LOGGER_INFO) << "Population successfully initialized." << endl;
}

void EACirc::saveProgress(const string stateFilename, const string populationFilename) {
    if (m_status != STAT_OK) return;
    saveState(stateFilename);
    savePopulation(populationFilename);
}

void EACirc::prepare() {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & EACIRC_CONFIG_LOADED) != EACIRC_CONFIG_LOADED) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }

    // PREPARE THE LOGGING FILES
    std::remove(FILE_BOINC_FRACTION_DONE);
    if (!m_settings.main.recommenceComputation) {
        std::remove(FILE_FITNESS_PROGRESS);
        std::remove(FILE_BEST_FITNESS);
        std::remove(FILE_AVG_FITNESS);
        std::remove(FILE_GALIB_SCORES);
        std::remove(FILE_TEST_VECTORS_HR);
    }

    // map net share for random data, if used (for METACENTRUM resources)
    if (m_settings.random.useNetShare) {
        mainLogger.out(LOGGER_INFO) << "Trying to map net share for random data." << endl;
        int errorCode = system("net use K: \\\\10.1.1.10\\boinc /u:boinc_smb_rw Wee5Eiw9");
        system("ping -n 10 127.0.0.1 > NUL");
        mainLogger.out(LOGGER_INFO) << "Mapping net share ended (error code: " << errorCode << ")." << endl;
    }

    // initialize project
    m_status = m_project->initializeProject();
    mainLogger.out(LOGGER_INFO) << "Project now fully initialized. (" << m_project->shortDescription() << ")" << endl;
    // initialize evaluator
    if (m_settings.main.evaluatorType < EVALUATOR_PROJECT_SPECIFIC_MINIMUM) {
        pGlobals->evaluator = IEvaluator::getStandardEvaluator(m_settings.main.evaluatorType);
    } else {
        pGlobals->evaluator = m_project->getProjectEvaluator();
    }
    if (pGlobals->evaluator != NULL) {
        mainLogger.out(LOGGER_INFO) << "Evaluator initialized (" << pGlobals->evaluator->shortDescription() << ")." << endl;
    } else {
        mainLogger.out(LOGGER_ERROR) << "Cannot initialize evaluator (" << m_settings.main.evaluatorType << ")." << endl;
    }

    if (m_status == STAT_OK) {
        m_readyToRun |= EACIRC_PREPARED;
    }
}

void EACirc::initializeState() {
    if (m_status != STAT_OK) return;
    if ((m_readyToRun & (EACIRC_CONFIG_LOADED | EACIRC_PREPARED)) !=
            (EACIRC_CONFIG_LOADED | EACIRC_PREPARED)) {
        m_status = STAT_CONFIG_SCRIPT_INCOMPLETE;
        return;
    }
    // load or create STATE
    if (m_settings.main.recommenceComputation) {
        loadState(FILE_STATE);
    } else {
        createState();
    }
    // create headers in test vector files
    m_status = m_project->createTestVectorFilesHeadersMain();
    if (m_status != STAT_OK) return;
    // load or create POPULATION
    if (m_settings.main.loadInitialPopulation) {
        loadPopulation(FILE_POPULATION);
    } else {
        m_status = m_project->generateAndSaveTestVectors();
        if (m_status == STAT_OK) {
            mainLogger.out(LOGGER_INFO) << "Initial test vectors generated." << endl;
        } else {
            mainLogger.out(LOGGER_ERROR) << "Initial test vectors generation failed." << endl;
            return;
        }
        createPopulation();
    }

    if (m_status == STAT_OK) {
        m_readyToRun |= EACIRC_INITIALIZED;
    }
}

void EACirc::seedAndResetGAlib(const GAPopulation &population) {
    // set GAlib seed
    GAResetRNG(m_currentGalibSeed);
    // init new instance of gentetic algorithm
    GASteadyStateGA* gaTemp = new GASteadyStateGA(population);
    // reset population stats
    gaTemp->pop->touch();
    // delete any previous instance of genetic algorithm
    if (m_gaData != NULL) {
        delete m_gaData;
        m_gaData = NULL;
    }
    m_gaData = gaTemp;
    // initialize the new genetic algorithm
    m_gaData->populationSize(m_settings.ga.popupationSize);
    m_gaData->nReplacement(2 * m_settings.ga.popupationSize / 3);
    m_gaData->nGenerations(m_settings.main.numGenerations);
    m_gaData->pCrossover(m_settings.ga.probCrossing);
    m_gaData->pMutation(m_settings.ga.probMutation);
    m_gaData->scoreFilename(FILE_GALIB_SCORES);
    m_gaData->scoreFrequency(1);	// keep the scores of every generation
    m_gaData->flushFrequency(1);	// specify how often to write the score to disk
    m_gaData->selectScores(GAStatistics::AllScores);
    mainLogger.out(LOGGER_INFO) << "GAlib seeded and reset." << endl;
}

void EACirc::evaluateStep() {
    if (m_status != STAT_OK) return;
    int totalGeneration = m_actGener + m_oldGenerations;
    pGlobals->stats.bestGenerFit = 0;
    GA1DArrayGenome<unsigned long> genome = (GA1DArrayGenome<unsigned long>&) m_gaData->population().best();

    float bestFit = CircuitGenome::Evaluator(genome);
    //float bestFit = genome.score();

    ofstream bestfitfile(FILE_BEST_FITNESS, ios::app);
    ofstream avgfitfile(FILE_AVG_FITNESS, ios::app);
    bestfitfile << totalGeneration << "," << bestFit << endl;
    if (pGlobals->stats.numAvgGenerFit > 0) avgfitfile << totalGeneration << "," << pGlobals->stats.avgGenerFit / pGlobals->stats.numAvgGenerFit << endl;
    else avgfitfile << totalGeneration << "," << "division by zero!!" << endl;
    bestfitfile.close();
    avgfitfile.close();

    ostringstream os2;
    os2 << "(" << totalGeneration << " gen.): " << pGlobals->stats.avgGenerFit << "/" << pGlobals->stats.numAvgGenerFit;
    os2 << " avg, " << bestFit << " best, avgPredict: " << pGlobals->stats.avgPredictions / pGlobals->stats.numAvgGenerFit;
    os2 << ", totalBest: " << pGlobals->stats.maxFit;
    string message = os2.str();
    // SAVE FITNESS PROGRESS
    ofstream out(FILE_FITNESS_PROGRESS, ios::app);
    out << message << endl;
    out.close();

    pGlobals->stats.clear();
    //pGACirc->avgGenerFit = 0;
    //pGACirc->numAvgGenerFit = 0;
    //pGACirc->avgPredictions = 0;
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

    // clear scores
    pGlobals->stats.clear();
    if (!pGlobals->settings->main.recommenceComputation) {
        std::remove(FILE_GALIB_SCORES);
    }

    int	changed = 1;
    bool evaluateNow = false;
    fstream fitfile;

    //GA1DArrayGenome<unsigned long> genomeBest(m_settings.circuit.genomeSize, CircuitGenome::Evaluator);
    //genomeBest = m_gaData->population().individual(0);

    mainLogger.out(LOGGER_INFO) << "Starting evolution." << endl;
    for (m_actGener = 1; m_actGener <= m_settings.main.numGenerations; m_actGener++) {
        pGlobals->testVectors.newSet = false;
        if (m_status != STAT_OK) {
            mainLogger.out(LOGGER_ERROR) << "Ooops, something went wrong, stopping. " << "(error: " << statusToString(m_status) << ")." << endl;
            break;
        }

        //FRACTION FILE FOR BOINC
        fitfile.open(FILE_BOINC_FRACTION_DONE, fstream::out | ios::trunc);
        fitfile << ((float)(m_actGener))/((float)(m_settings.main.numGenerations));
        fitfile.close();

        // DO NOT EVOLVE.. (if evolution is off)
        if (m_settings.ga.evolutionOff) {
            m_status = m_project->generateAndSaveTestVectors();
            evaluateStep();
            continue;
        }

        // GENERATE TEST VECTORS IF NEEDED
        if (m_settings.testVectors.setChangeProgressive) {
            // TODO: understand and correct
            if (changed > m_actGener/m_settings.testVectors.setChangeFrequency + 1) {
                m_status = m_project->generateAndSaveTestVectors();
                evaluateNow = true;
                changed = 0;
            }
        } else {
            if (m_actGener %(m_settings.testVectors.setChangeFrequency) == 1) {
                m_status = m_project->generateAndSaveTestVectors();
                if (m_status == STAT_OK) {
                    mainLogger.out(LOGGER_INFO) << "Test vectors regenerated." << endl;
                }
            }
            if ( m_settings.testVectors.evaluateBeforeTestVectorChange &&
                 m_actGener %(m_settings.testVectors.setChangeFrequency) == 0) {
                evaluateNow = true;
            }
            if (!m_settings.testVectors.evaluateBeforeTestVectorChange &&
                 m_actGener % (m_settings.testVectors.setChangeFrequency) == 1) {
                evaluateNow = true;
            }
        }
        // variable for computing when TVCGProgressive = true
        changed++;

        // RESET EVALUTION FOR ALL GENOMS
        m_gaData->pop->flushEvalution();
        // GA evolution step
        m_gaData->step();

        //genomeBest = (GA1DArrayGenome<unsigned long>&) m_gaData->population().best();// .statistics().bestIndividual();

        if (evaluateNow || m_settings.testVectors.evaluateEveryStep) {
            evaluateStep();
            evaluateNow = false;
        }

        // if needed, reseed GAlib and save state and population
        if (m_settings.main.saveStateFrequency != 0
                && m_actGener % m_settings.main.saveStateFrequency == 0) {
            galibGenerator->getRandomFromInterval(ULONG_MAX,&m_currentGalibSeed);
            seedAndResetGAlib(m_gaData->population());
            saveProgress(FILE_STATE,FILE_POPULATION);
        }
    }

    // commented for testing purposes of saving state

    // GENERATE FRESH NEW SET AND EVALUATE THE RESULT
    //m_status = m_evaluator->generateAndSaveTestVectors();
    //m_evaluator->evaluateStep(genomeBest, m_actGener);
    GA1DArrayGenome<unsigned long> genomeBest = (GA1DArrayGenome<unsigned long>&) m_gaData->population().best();// .statistics().bestIndividual();
    //Print the best circuit
    CircuitGenome::PrintCircuit(genomeBest,FILE_BEST_CIRCUIT,0,1);
    //m_status = saveState(FILE_STATE,FILE_POPULATION);
}
