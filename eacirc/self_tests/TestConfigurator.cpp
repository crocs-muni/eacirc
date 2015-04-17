#include "TestConfigurator.h"
#include <iostream>
#include <Catch.h>
#include "EACirc.h"
#include <tinyxml.h>

TestConfigurator::TestConfigurator()
    : m_currentProject(0) {
    m_projects.push(PROJECT_ESTREAM);
    m_projects.push(PROJECT_SHA3);
    // can we open files set in PROJECT_FILE_DISTINGUISHER?
    string conf = IProject::getTestingConfiguration(PROJECT_FILE_DISTINGUISHER);
    TiXmlDocument doc(string("configuration").c_str());
    doc.Parse(conf.c_str());
    TiXmlHandle hDoc(&doc);
    TiXmlElement* pElem=hDoc.FirstChildElement().Element();
    if (pElem != NULL) {
        TiXmlNode* pRoot = pElem->Clone();
        string filename1 = getXMLElementValue(pRoot,"FILENAME_1");
        string filename2 = getXMLElementValue(pRoot,"FILENAME_2");
        ifstream file1(filename1);
        ifstream file2(filename2);
        if (file1.is_open() && file2.is_open()) {
            m_projects.push(PROJECT_FILE_DISTINGUISHER);
            file1.close();
            file2.close();
        } else {
            WARN(string("######## Project ")+CommonFnc::toString(PROJECT_FILE_DISTINGUISHER)+"cannot be tested ########");
            WARN(string("######## Could not open files ")+filename1+", "+filename2+" ########");
        }
    }
}

TestConfigurator::TestConfigurator(int projectType)
    : m_currentProject(0) {
    m_projects.push(projectType);
}

TestConfigurator::~TestConfigurator() {}

bool TestConfigurator::nextProject() {
    if (m_projects.empty()) {
        return false;
    }
    m_currentProject = m_projects.front();
    m_projects.pop();
    WARN("########");
    WARN(string("######## Testing project ")+CommonFnc::toString(m_currentProject)+" ########");
    WARN("########");
    return true;
}

int TestConfigurator::getCurrentProject() {
    return m_currentProject;
}

void TestConfigurator::compareFilesByLine(string filename1, string filename2) const {
    string line1,line2;
    ifstream file1(filename1);
    ifstream file2(filename2);
    if (!file1.is_open() || !file2.is_open()) {
        WARN("Could not open files " << filename1 << " and " << filename2 << ".");
        return;
    }
    int differCount = 0;
    while ((!file1.eof() || !file2.eof()) && differCount <=5 ) {
        getline(file1,line1);
        getline(file2,line2);
        if (line1 != line2) {
            differCount++;
        }
        SCOPED_INFO("Comparing files " << filename1 << " and " << filename2 << ".");
        CHECK(line1 == line2);
    }
    if (differCount > 5) {
        WARN("Given files (" << filename1 << ", " << filename2 << " differ in more than 6 lines!");
    }
    file1.close();
    file2.close();
}

void TestConfigurator::backupFile(string filename) {
    string backupFilename = filename + BACKUP_SUFFIX;
    CommonFnc::removeFile(backupFilename.c_str());
    CHECK(rename(filename.c_str(),backupFilename.c_str()) == 0);
}

void TestConfigurator::backupResults() {
    backupFile(FILE_GALIB_SCORES);
    backupFile(FILE_FITNESS_PROGRESS);
    backupFile(FILE_BEST_FITNESS);
    backupFile(FILE_AVG_FITNESS);
    backupFile(FILE_STATE);
    backupFile(FILE_POPULATION);
}

void TestConfigurator::compareResults() const {
    compareFilesByLine(FILE_GALIB_SCORES,string(FILE_GALIB_SCORES)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_FITNESS_PROGRESS,string(FILE_FITNESS_PROGRESS)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_BEST_FITNESS,string(FILE_BEST_FITNESS)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_AVG_FITNESS,string(FILE_AVG_FITNESS)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_STATE,string(FILE_STATE)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_POPULATION,string(FILE_POPULATION)+BACKUP_SUFFIX);
}

void TestConfigurator::runEACirc() const {
    WARN("######## Running EACirc ########");
    mainLogger.out(LOGGER_INFO) << "Configuration file: "  << FILE_CONFIG << endl;
    EACirc eacirc;
    eacirc.loadConfiguration(FILE_CONFIG);
    eacirc.prepare();
    eacirc.initializeState();
    eacirc.run();
    WARN("######## Ending EACirc (status: " << statusToString(eacirc.getStatus()) << " ) ########");
    CHECK(eacirc.getStatus() == STAT_OK);
}

void TestConfigurator::prepareConfiguration(int projectType) const {
    string projectConfiguration = IProject::getTestingConfiguration(projectType);
    string configuration;
    configuration += "<?xml version=\"1.0\" encoding=\"UTF-8\" ?><EACIRC>";
    configuration += mainConfiguration;
    configuration += projectConfiguration;
    configuration += "</EACIRC>";
    ofstream configFile(FILE_CONFIG);
    REQUIRE(configFile.is_open());
    configFile << configuration;
    configFile.close();
    // set correct project constant
    TiXmlNode* pRootConfig = NULL;
    REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/PROJECT",CommonFnc::toString(projectType)) == STAT_OK);
    REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    pRootConfig = NULL;
}

void TestConfigurator::prepareConfiguration() const {
    prepareConfiguration(m_currentProject);
}

string TestConfigurator::mainConfiguration =
        "<NOTES>self-test configuration</NOTES>"
        "<MAIN>"
        "    <CIRCUIT_REPRESENTATION>1</CIRCUIT_REPRESENTATION>"
        "    <PROJECT>100</PROJECT>"
        "    <EVALUATOR>26</EVALUATOR>"
        "    <EVALUATOR_PRECISION>10</EVALUATOR_PRECISION>"
        "    <RECOMMENCE_COMPUTATION>0</RECOMMENCE_COMPUTATION>"
        "    <LOAD_INITIAL_POPULATION>0</LOAD_INITIAL_POPULATION>"
        "    <NUM_GENERATIONS>20</NUM_GENERATIONS>"
        "    <SAVE_STATE_FREQ>10</SAVE_STATE_FREQ>"
        "    <CIRCUIT_SIZE_INPUT>16</CIRCUIT_SIZE_INPUT>"
        "    <CIRCUIT_SIZE_OUTPUT>2</CIRCUIT_SIZE_OUTPUT>"
        "</MAIN>"
        "<OUTPUTS>"
        "    <GRAPH_FILES>1</GRAPH_FILES>"
        "    <INTERMEDIATE_CIRCUITS>1</INTERMEDIATE_CIRCUITS>"
        "    <ALLOW_PRUNNING>0</ALLOW_PRUNNING>"
        "    <SAVE_TEST_VECTORS>0</SAVE_TEST_VECTORS>"
        "</OUTPUTS>"
        "<RANDOM>"
        "    <USE_FIXED_SEED>1</USE_FIXED_SEED>"
        "    <SEED>123456789</SEED>"
        "    <BIAS_RNDGEN_FACTOR>95</BIAS_RNDGEN_FACTOR>"
        "    <USE_NET_SHARE>0</USE_NET_SHARE>"
        "    <QRNG_PATH>../../qrng/;/mnt/centaur/home/eacirc/qrng/;C:/RNG/;D:/RandomData/</QRNG_PATH>"
        "    <QRNG_MAX_INDEX>192</QRNG_MAX_INDEX>"
        "</RANDOM>"
        "<CUDA>"
        "    <ENABLED>0</ENABLED>"
        "    <SOMETHING>something</SOMETHING>"
        "</CUDA>"
        "<GA>"
        "    <EVOLUTION_OFF>0</EVOLUTION_OFF>"
        "    <PROB_MUTATION>0.05</PROB_MUTATION>"
        "    <MUTATE_FUNCTIONS>1</MUTATE_FUNCTIONS>"
        "    <MUTATE_CONNECTORS>1</MUTATE_CONNECTORS>"
        "    <PROB_CROSSING>0.5</PROB_CROSSING>"
        "    <POPULATION_SIZE>20</POPULATION_SIZE>"
        "    <REPLACEMENT_SIZE>13</REPLACEMENT_SIZE>"
        "</GA>"
        "<GATE_CIRCUIT>"
        "    <NUM_LAYERS>5</NUM_LAYERS>"
        "    <SIZE_LAYER>8</SIZE_LAYER>"
        "    <NUM_CONNECTORS>4</NUM_CONNECTORS>"
        "    <USE_MEMORY>0</USE_MEMORY>"
        "    <SIZE_MEMORY>2</SIZE_MEMORY>"
        "    <ALLOWED_FUNCTIONS>"
        "        <FNC_NOP>1</FNC_NOP>"
        "        <FNC_CONS>1</FNC_CONS>"
        "        <FNC_AND>1</FNC_AND>"
        "        <FNC_NAND>1</FNC_NAND>"
        "        <FNC_OR>1</FNC_OR>"
        "        <FNC_XOR>1</FNC_XOR>"
        "        <FNC_NOR>1</FNC_NOR>"
        "        <FNC_NOT>1</FNC_NOT>"
        "        <FNC_SHIL>1</FNC_SHIL>"
        "        <FNC_SHIR>1</FNC_SHIR>"
        "        <FNC_ROTL>1</FNC_ROTL>"
        "        <FNC_ROTR>1</FNC_ROTR>"
        "        <FNC_EQ>1</FNC_EQ>"
        "        <FNC_LT>1</FNC_LT>"
        "        <FNC_GT>1</FNC_GT>"
        "        <FNC_LEQ>1</FNC_LEQ>"
        "        <FNC_GEQ>1</FNC_GEQ>"
        "        <FNC_BSLC>1</FNC_BSLC>"
        "        <FNC_READ>1</FNC_READ>"
        "        <FNC_JVM>0</FNC_JVM>"
        "    </ALLOWED_FUNCTIONS>"
        "</GATE_CIRCUIT>"
        "<POLYNOMIAL_CIRCUIT>"
        "    <NUM_POLYNOMIALS>1</NUM_POLYNOMIALS>"
        "    <MUTATE_TERM_STRATEGY>0</MUTATE_TERM_STRATEGY>"
        "    <MAX_TERMS>50</MAX_TERMS>"
        "    <TERM_COUNT_P>0.70</TERM_COUNT_P>"
        "    <TERM_VAR_P>0.60</TERM_VAR_P>"
        "    <ADD_TERM_P>0.05</ADD_TERM_P>"
        "    <ADD_TERM_STRATEGY>0</ADD_TERM_STRATEGY>"
        "    <RM_TERM_P>0.05</RM_TERM_P>"
        "    <RM_TERM_STRATEGY>0</RM_TERM_STRATEGY>"
        "    <CROSSOVER_RANDOMIZE_POLY>1</CROSSOVER_RANDOMIZE_POLY>"
        "    <CROSSOVER_TERM_P>0.1</CROSSOVER_TERM_P>"
        "</POLYNOMIAL_CIRCUIT>"
        "<TEST_VECTORS>"
        "    <INPUT_LENGTH>16</INPUT_LENGTH>"
        "    <OUTPUT_LENGTH>2</OUTPUT_LENGTH>"
        "    <SET_SIZE>1000</SET_SIZE>"
        "    <SET_CHANGE_FREQ>5</SET_CHANGE_FREQ>"
        "    <EVALUATE_BEFORE_TEST_VECTOR_CHANGE>0</EVALUATE_BEFORE_TEST_VECTOR_CHANGE>"
        "    <EVALUATE_EVERY_STEP>0</EVALUATE_EVERY_STEP>"
        "</TEST_VECTORS>";
