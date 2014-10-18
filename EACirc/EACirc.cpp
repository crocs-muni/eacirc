#include "EACirc.h"
#include "CommonFnc.h"
#include "generators/BiasRndGen.h"
#include "generators/QuantumRndGen.h"
#include "generators/MD5RndGen.h"
#include "garandom.h"
#include "XMLProcessor.h"

#ifdef _WIN32
	#include <Windows.h>
	#define getpid() GetCurrentProcessId()
#endif
#ifdef __linux__
	#include <sys/types.h>
	#include <unistd.h>
#endif

EACirc::EACirc()
    : m_status(STAT_OK), m_originalSeed(0), m_currentGalibSeed(0), m_circuit(NULL), m_project(NULL), m_gaData(NULL),
      m_readyToRun(0), m_actGener(0), m_oldGenerations(0), m_evaluateStepVisitor(NULL) {
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
        pGlobals->stats.release();
        delete pGlobals;
    }
    pGlobals = NULL;
    if (rndGen) delete rndGen;
    rndGen = NULL;
    if (biasRndGen) delete biasRndGen;
    biasRndGen = NULL;
    if (mainGenerator) delete mainGenerator;
    mainGenerator = NULL;
    if (m_circuit) delete m_circuit;
    m_circuit = NULL;
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

    // load circuit representation and its configuration
    m_circuit = ICircuit::getCircuit(m_settings.main.circuitType);
    if (m_circuit == NULL) {
        m_status = STAT_CIRCUIT_BACKEND_ERROR;
        mainLogger.out(LOGGER_ERROR) << "Could not load circuit representation." << endl;
        return;
    }
    m_status = m_circuit->loadCircuitConfiguration(pRoot);
    if (m_status != STAT_OK) return;
    mainLogger.out(LOGGER_INFO) << "Circuit representation configuration loaded. (" << m_circuit->shortDescription() << ")" << endl;

    // load project and its configuration
    m_project = IProject::getProject(m_settings.main.projectType);
    if (m_project == NULL) {
        m_status = STAT_PROJECT_ERROR;
        mainLogger.out(LOGGER_ERROR) << "Could not load project." << endl;
        return;
    }
    m_status = m_project->loadProjectConfiguration(pRoot);
    if (m_status != STAT_OK) return;
    mainLogger.out(LOGGER_INFO) << "Project configuration loaded. (" << m_project->shortDescription() << ")" << endl;

    // allocate space for testVecotrs
    pGlobals->testVectors.allocate();
    // allocate statistics
    pGlobals->stats.allocate();

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
    if (m_settings.ga.replacementSize <= 0 || m_settings.ga.replacementSize > m_settings.ga.popupationSize) {
        mainLogger.out(LOGGER_ERROR) << "Replacement size must be greater than 0 and must not exceed population size." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (m_settings.testVectors.inputLength < m_settings.main.circuitSizeInput) {
        mainLogger.out(LOGGER_ERROR) << "Test vector input length is smaller than circuit input layer." << endl;
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

    pElem = new TiXmlElement("pvalues_best_individual");
    ostringstream pvalues;
    for (unsigned int i = 0; i < pGlobals->stats.pvaluesBestIndividual->size(); i++) {
        pvalues << pGlobals->stats.pvaluesBestIndividual->at(i) << " ";
    }
    pElem->LinkEndChild(new TiXmlText(pvalues.str().c_str()));
    pRoot->LinkEndChild(pElem);

    pElem = new TiXmlElement("random_generators");
    pElem2 = new TiXmlElement("main_generator");
    pElem2->LinkEndChild(mainGenerator->exportGenerator());
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
    // restore pvalues of best individuals
    istringstream pvalues(getXMLElementValue(pRoot,"pvalues_best_individual"));
    double val = 0;
    while (pvalues >> val) {
        pGlobals->stats.pvaluesBestIndividual->push_back(val);
    }

    // initialize random generators (main, quantum, bias)
    mainGenerator = IRndGen::parseGenerator(getXMLElement(pRoot,"random_generators/main_generator/generator"));
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
    // INIT RNG
    mainGenerator->getRandomFromInterval(ULONG_MAX,&generatorSeed);
    rndGen = new QuantumRndGen(generatorSeed, m_settings.random.qrngPath);
    mainLogger.out(LOGGER_INFO) << "Random generator initialized (" << rndGen->shortDescription() << ")" << endl;
    // INIT BIAS RNDGEN
    mainGenerator->getRandomFromInterval(ULONG_MAX,&generatorSeed);
    biasRndGen = new BiasRndGen(generatorSeed, m_settings.random.qrngPath, m_settings.random.biasRndGenFactor);
    mainLogger.out(LOGGER_INFO) << "Bias random generator initialized (" << biasRndGen->shortDescription() << ")" << endl;

    // GENERATE SEED FOR GALIB
    mainGenerator->getRandomFromInterval(UINT_MAX,&m_currentGalibSeed);
    mainLogger.out(LOGGER_INFO) << "State successfully initialized." << endl;
    // INIT PROJECT STATE
    m_project->initializeProjectState();
    mainLogger.out(LOGGER_INFO) << "Project intial state setup successful (" << m_project->shortDescription() << ")." << endl;
}

void EACirc::savePopulation(const string filename) {
    TiXmlElement* pRoot = m_circuit->io()->populationHeader(m_settings.ga.popupationSize);
    TiXmlElement* pElem = NULL;
    TiXmlElement* pElem2 = NULL;

    pElem = new TiXmlElement("population");
    string textCircuit;
    for (int i = 0; i < m_settings.ga.popupationSize; i++) {
        // note: it is not necessary to call individual i in SCALED order
        //       however then the population files differ in order ('diff' cannot be used to finding bugs)
        GAGenome & genome = m_gaData->population().individual(i,GAPopulation::SCALED);
        m_status = m_circuit->io()->genomeToBinary(genome ,textCircuit);
        if (m_status != STAT_OK) {
            mainLogger.out(LOGGER_ERROR) << "Could not save genome in population to file " << filename << "." << endl;
            delete pElem;
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
    if (m_settings.gateCircuit.numLayers != settingsValue) {
        mainLogger.out(LOGGER_ERROR) << "Cannot load population - incompatible number of layers (";
        mainLogger.out() << m_settings.gateCircuit.numLayers << " vs. " << settingsValue << ")." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    settingsValue = atoi(getXMLElementValue(pRoot,"circuit_dimensions/size_layer").c_str());
    if (m_settings.gateCircuit.sizeLayer != settingsValue) {
        mainLogger.out(LOGGER_ERROR) << "Cannot load population - incompatible layer size (";
        mainLogger.out() << m_settings.gateCircuit.sizeLayer << " vs. " << settingsValue << ")." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    settingsValue = atoi(getXMLElementValue(pRoot,"circuit_dimensions/size_input_layer").c_str());
    if (m_settings.main.circuitSizeInput != settingsValue) {
        mainLogger.out(LOGGER_ERROR) << "Cannot load population - incompatible input layer size (";
        mainLogger.out() << m_settings.main.circuitSizeInput << " vs. " << settingsValue << ")." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    settingsValue = atoi(getXMLElementValue(pRoot,"circuit_dimensions/size_output_layer").c_str());
    if (m_settings.main.circuitSizeOutput != settingsValue) {
        mainLogger.out(LOGGER_ERROR) << "Cannot load population - incompatible output layer size (";
        mainLogger.out() << m_settings.main.circuitSizeOutput << " vs. " << settingsValue << ")." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    settingsValue = atoi(getXMLElementValue(pRoot,"circuit_dimensions/size_memory").c_str());
    if (m_settings.gateCircuit.sizeMemory != settingsValue) {
        mainLogger.out(LOGGER_ERROR) << "Cannot load population - incompatible memory size (";
        mainLogger.out() << m_settings.gateCircuit.sizeMemory << " vs. " << settingsValue << ")." << endl;
        m_status = STAT_CONFIG_INCORRECT;
    }
    if (m_status != STAT_OK) {
        delete pRoot;
        return;
    }
    
    GAPopulation * population = new GAPopulation;
    GAGenome * genome = m_circuit->createGenome(true);
    
    // LOAD genomes
    TiXmlElement* pGenome = getXMLElement(pRoot,"population/genome")->ToElement();
    string textCircuit;
    for (int i = 0; i < savedPopulationSize; i++) {
        if (pGenome->GetText() == NULL) {
            mainLogger.out(LOGGER_ERROR) << "Too few genomes in population - expecting " << savedPopulationSize << "." << endl;
            m_status = STAT_DATA_CORRUPTED;
            delete pRoot;
            delete genome;
            delete population;
            return;
        }
        textCircuit = pGenome->GetText();
        m_status = m_circuit->io()->genomeFromBinary(textCircuit, *genome);
        if (m_status != STAT_OK) return;
        population->add(*genome);
        pGenome = pGenome->NextSiblingElement();
    }

    seedAndResetGAlib(*population);
    delete pRoot;
    delete genome;
    delete population;
    mainLogger.out(LOGGER_INFO) << "Population successfully loaded from file " << filename << "." << endl;
}

void EACirc::createPopulation() {
    if (m_status != STAT_OK) return;
    // seed GAlib (initializations may require random numbers)
    GAResetRNG(m_currentGalibSeed);

    // Create a basic genome used for this problem.
    GAPopulation * population = m_circuit->createPopulation();
    // create genetic algorithm and initialize population
    seedAndResetGAlib(*population);
    delete population;
    population = NULL;
    
    mainLogger.out(LOGGER_INFO) << "Initializing population, representation: " << m_circuit->shortDescription() << endl;
    m_gaData->initialize();
    
    // reset GAlib seed
    mainGenerator->getRandomFromInterval(UINT_MAX, &m_currentGalibSeed);
    seedAndResetGAlib(m_gaData->population());

    mainLogger.out(LOGGER_INFO) << "Population successfully initialized: " << m_gaData->population().className() << endl;
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

    // prepare files for logging
    removeFile(FILE_BOINC_FRACTION_DONE);
    if (!m_settings.main.recommenceComputation) {
        removeFile(FILE_FITNESS_PROGRESS);
        removeFile(FILE_BEST_FITNESS);
        removeFile(FILE_AVG_FITNESS);
        removeFile(FILE_GALIB_SCORES);
        removeFile(FILE_TEST_VECTORS_HR);
        removeFile(FILE_HISTOGRAMS);
        ofstream fitnessProgressFile(FILE_FITNESS_PROGRESS, ios_base::trunc);
        fitnessProgressFile << "Fitness statistics for selected generations" << endl;
        for (int i = 0; i < log(pGlobals->settings->main.numGenerations)/log(10) - 3; i++) fitnessProgressFile << " ";
        fitnessProgressFile << "gen\tavg";
        for (int i = 0; i < FITNESS_PRECISION_LOG +2 - 3; i++) fitnessProgressFile << " ";
        fitnessProgressFile << "\tmax";
        for (int i = 0; i < FITNESS_PRECISION_LOG +2 - 3; i++) fitnessProgressFile << " ";
        fitnessProgressFile << "\tmin";
        for (int i = 0; i < FITNESS_PRECISION_LOG +2 - 3; i++) fitnessProgressFile << " ";
        fitnessProgressFile << "\tpnm";
        for (int i = 0; i < FITNESS_PRECISION_LOG +2 - 3; i++) fitnessProgressFile << " ";
        fitnessProgressFile << "\tpvl" << endl;
        fitnessProgressFile.close();
    }

    // map net share for random data, if used (for METACENTRUM resources)
    if (m_settings.random.useNetShare) {
        mainLogger.out(LOGGER_INFO) << "Trying to map net share for random data." << endl;
        int errorCode = system("net use K: \\\\10.1.1.10\\boinc /u:boinc_smb_rw Wee5Eiw9");
        system("ping -n 10 127.0.0.1 > NUL");
        mainLogger.out(LOGGER_INFO) << "Mapping net share ended (error code: " << errorCode << ")." << endl;
    }

    // initialize backend
    m_status = m_circuit->initialize();
    if (m_status != STAT_OK) return;
    mainLogger.out(LOGGER_INFO) << "Circuit backend now fully initialized. (" << m_circuit->shortDescription() << ")" << endl;

    // initialize project
    m_status = m_project->initializeProject();
    if (m_status != STAT_OK) return;
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
        m_status = STAT_CONFIG_INCORRECT;
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
    m_gaData->nReplacement(m_settings.ga.replacementSize);
    m_gaData->nGenerations(m_settings.main.numGenerations);
    m_gaData->pCrossover(m_settings.ga.probCrossing);
    m_gaData->pMutation(m_settings.ga.probMutation);
    // cannot disable scaling, some evaluators produce fitness values that are not usable directly
    //GANoScaling scaler;
    //m_gaData->scaling(scaler);
    m_gaData->scoreFilename(FILE_GALIB_SCORES);
    m_gaData->scoreFrequency(1);	// keep the scores of every generation
    m_gaData->flushFrequency(1);	// specify how often to write the score to disk
    m_gaData->selectScores(GAStatistics::AllScores);
    mainLogger.out(LOGGER_INFO) << "GAlib seeded and reset." << endl;
}

void EACirc::preEvaluate() {
    // add fitness of the best individual to statistics vector
    if (m_gaData->population().evaluated) {
        GAGenome & bestGenome = m_gaData->population().best();
        pGlobals->stats.pvaluesBestIndividual->push_back(bestGenome.evaluate(gaTrue));
    } else {    // we just loaded population, use the first one (population is saved sorted)
        GAGenome & bestGenome = m_gaData->population().individual(0);
        pGlobals->stats.pvaluesBestIndividual->push_back(bestGenome.evaluate(gaTrue));
    }

}

void EACirc::evaluateStep() {
    if (m_status != STAT_OK) return;
    int totalGeneration = m_actGener + m_oldGenerations;

    // add scores to fitness progress file
    ofstream fitProgressFile(FILE_FITNESS_PROGRESS, ios_base::app);
    int digitsInGeneration = max<int>(log(pGlobals->settings->main.numGenerations) / log(10) + 1, 3); // header text "gen" is 3 chars long
    fitProgressFile << setw(digitsInGeneration) << right << totalGeneration;
    fitProgressFile << left << setprecision(FITNESS_PRECISION_LOG) << fixed;
    fitProgressFile << "\t" << m_gaData->statistics().current(GAStatistics::Mean);
    fitProgressFile << "\t" << m_gaData->statistics().current(GAStatistics::Maximum);
    fitProgressFile << "\t" << m_gaData->statistics().current(GAStatistics::Minimum);
    fitProgressFile << "\t" << pGlobals->stats.pvaluesBestIndividual->size();
    
    if (pGlobals->stats.pvaluesBestIndividual->size() > 0){
        fitProgressFile << "\t" << pGlobals->stats.pvaluesBestIndividual->back();
    } else {
        fitProgressFile << "\t" << -1;
    }
            
    fitProgressFile << endl;
    fitProgressFile.close();

    // add scores to graph files
    if (pGlobals->settings->outputs.graphFiles) {
        ofstream bestFitFile(FILE_BEST_FITNESS, ios_base::app);
        ofstream avgFitFile(FILE_AVG_FITNESS, ios_base::app);
        bestFitFile << totalGeneration << ", " << m_gaData->statistics().current(GAStatistics::Maximum) << endl;
        avgFitFile << totalGeneration << ", " << m_gaData->statistics().current(GAStatistics::Mean) << endl;
        bestFitFile.close();
        avgFitFile.close();
    }

    // print currently best circuit
    if (pGlobals->settings->outputs.intermediateCircuits) {
        GAGenome & bestGenome = m_gaData->population().best();
        ostringstream fileName;
        fileName << FILE_CIRCUIT_PREFIX << "g" << totalGeneration << "_";
        fileName << setprecision(FILE_CIRCUIT_PRECISION) << fixed << m_gaData->statistics().current(GAStatistics::Maximum);
        string filePath = fileName.str();
        m_circuit->io()->outputGenomeFiles(bestGenome, filePath);
    }

    // save generation stats for total scores
    pGlobals->stats.avgAvgFitSum += m_gaData->statistics().current(GAStatistics::Mean);
    pGlobals->stats.avgMinFitSum += m_gaData->statistics().current(GAStatistics::Minimum);
    pGlobals->stats.avgMaxFitSum += m_gaData->statistics().current(GAStatistics::Maximum);
    pGlobals->stats.avgCount++;
    
    // Call visitor, if non-null.
    if (m_evaluateStepVisitor!=NULL){
        m_evaluateStepVisitor(this);
    }
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

    // clear galib score file
    if (!pGlobals->settings->main.recommenceComputation) {
        removeFile(FILE_GALIB_SCORES);
    }

    bool evaluateNow = false;
    fstream fitfile;

    mainLogger.out(LOGGER_INFO) << "Starting evolution." << endl;
    for (m_actGener = 1; m_actGener <= m_settings.main.numGenerations; m_actGener++) {
        pGlobals->stats.actGener = m_actGener;

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
            preEvaluate();
            m_gaData->pop->flushEvalution();
            m_gaData->pop->evaluate(gaTrue);
            m_gaData->pop->scale(gaTrue);
            m_gaData->stats.update(m_gaData->population());
            evaluateStep();
            continue;
        }

        // GENERATE TEST VECTORS IF NEEDED
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

        // perform pre-evaluation, if needed
        if (evaluateNow || m_settings.testVectors.evaluateEveryStep) {
            preEvaluate();
        }

        // force re-evaluation if the test set is fresh
        if (pGlobals->testVectors.newSet) {
            m_gaData->pop->flushEvalution();
        }

        // evaluate population on new  and save statistics, if needed
        if (evaluateNow || m_settings.testVectors.evaluateEveryStep) {
            preEvaluate();
            m_gaData->pop->evaluate(gaTrue);
            m_gaData->pop->scale(gaTrue);
            m_gaData->stats.update(m_gaData->population());
            evaluateStep();
            evaluateNow = false;
        }

        // perform GA evolution step
        m_gaData->step();

        // if needed, reseed GAlib and save state and population
        if (m_settings.main.saveStateFrequency != 0
                && m_actGener % m_settings.main.saveStateFrequency == 0) {
            mainGenerator->getRandomFromInterval(UINT_MAX,&m_currentGalibSeed);
            seedAndResetGAlib(m_gaData->population());
            saveProgress(FILE_STATE,FILE_POPULATION);
        }
    }

    // output AvgAvg, AvgMax, AvgMin to logger
    mainLogger.out(LOGGER_INFO) << "Cumulative results for this run:" << endl << setprecision(FITNESS_PRECISION_LOG);
    mainLogger.out(LOGGER_INFO) << "   AvgAvg: " << pGlobals->stats.avgAvgFitSum / (double) pGlobals->stats.avgCount << endl;
    mainLogger.out(LOGGER_INFO) << "   AvgMax: " << pGlobals->stats.avgMaxFitSum / (double) pGlobals->stats.avgCount << endl;
    mainLogger.out(LOGGER_INFO) << "   AvgMin: " << pGlobals->stats.avgMinFitSum / (double) pGlobals->stats.avgCount << endl;

    // Kolmogorov-Smirnov test for the p-values uniformity.
    const unsigned long pvalsSize = pGlobals->stats.pvaluesBestIndividual->size();
    if (pvalsSize > 2){
        mainLogger.out(LOGGER_INFO) << "KS test on p-values, size=" << pvalsSize << endl;
        
        double KS_critical_alpha_5 = KS_get_critical_value(pvalsSize);
        double KS_P_value = KS_uniformity_test(pGlobals->stats.pvaluesBestIndividual);
        mainLogger.out(LOGGER_INFO) << "   KS Statistics: " << KS_P_value << endl;
        mainLogger.out(LOGGER_INFO) << "   KS critical value 0.05: " << KS_critical_alpha_5 << endl;
        
        if(KS_P_value > KS_critical_alpha_5) {
            mainLogger.out(LOGGER_INFO) << "   KS is in 5% interval -> uniformity hypothesis rejected." << endl;
        } else {
            mainLogger.out(LOGGER_INFO) << "   KS is not in 5% interval -> is uniform." << endl;
        }
    }
    
    // print the best circuit into separate file, prune if allowed
    GAGenome & genomeBest = m_gaData->population().best();
    m_circuit->io()->outputGenomeFiles(genomeBest, FILE_CIRCUIT_DEFAULT);
    GAGenome genomeProccessed = genomeBest;
    if (m_circuit->postProcess(genomeBest, genomeProccessed)) {
        m_circuit->io()->outputGenomeFiles(genomeProccessed, string(FILE_CIRCUIT_DEFAULT) + FILE_POSTPROCCESSED_SUFFIX);
    }
}
