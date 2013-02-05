#include <iostream>

#include "Tests.h"
#include "XMLProcessor.h"
#include "EACirc.h"

void compareFilesByLine(string filename1, string filename2) {
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

int backupFile(string filename) {
    string backupFilename = filename + BACKUP_SUFFIX;
    remove(backupFilename.c_str());
    if (rename(filename.c_str(),backupFilename.c_str()) != 0) {
        return STAT_FILE_WRITE_FAIL;
    }
    return STAT_OK;
}

void backupResults() {
    CHECK(backupFile(FILE_GALIB_SCORES) == STAT_OK);
    CHECK(backupFile(FILE_FITNESS_PROGRESS) == STAT_OK);
    CHECK(backupFile(FILE_BEST_FITNESS) == STAT_OK);
    CHECK(backupFile(FILE_AVG_FITNESS) == STAT_OK);
    CHECK(backupFile(FILE_STATE) == STAT_OK);
    CHECK(backupFile(FILE_POPULATION) == STAT_OK);
}

void compareResults() {
    compareFilesByLine(FILE_GALIB_SCORES,string(FILE_GALIB_SCORES)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_FITNESS_PROGRESS,string(FILE_FITNESS_PROGRESS)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_BEST_FITNESS,string(FILE_BEST_FITNESS)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_AVG_FITNESS,string(FILE_AVG_FITNESS)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_STATE,string(FILE_STATE)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_POPULATION,string(FILE_POPULATION)+BACKUP_SUFFIX);
}

int runEACirc() {
	if (mainLogger.getLogging()) {
		WARN("######## Running EACirc ########");
	}
    EACirc eacirc;
    eacirc.loadConfiguration(FILE_CONFIG);
    eacirc.prepare();
    eacirc.initializeState();
    eacirc.run();
	if (mainLogger.getLogging()) {
		WARN("######## Ending EACirc (error: " << eacirc.getStatus() << " ) ########");
	}
    return eacirc.getStatus();
}

TEST_CASE("xml/xpath","using simple variation of xpath to get/set element and attribute values in XML") {
    string location = "INFO/VERSION";
    TiXmlNode* pRoot = NULL;
    REQUIRE(loadXMLFile(pRoot,FILE_CONFIG) == STAT_OK);
    CHECK(getXMLElementValue(pRoot,location) == "5.0");

    string newData = "new data here!";
    REQUIRE(setXMLElementValue(pRoot,location,newData) == STAT_OK);
    string attrName = "TEST_ATTR";
    string attrValue = "1234";
    REQUIRE(setXMLElementValue(pRoot,location + "/@" + attrName,attrValue) == STAT_OK);
    REQUIRE(saveXMLFile(pRoot,FILE_CONFIG) == STAT_OK);

    REQUIRE(loadXMLFile(pRoot,FILE_CONFIG) == STAT_OK);
    CHECK(getXMLElementValue(pRoot,location) == newData);
    CHECK(getXMLElementValue(pRoot,location+"/@"+attrName) == attrValue);
    delete pRoot;
}

TEST_CASE("determinism/seed","testing whether run with random seed and second run with the same seed are same") {
    // general preparations
    REQUIRE(basicConfiguration::estream() == STAT_OK);
    TiXmlNode* pRootConfig = NULL;
    // prepare run 1
    REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","0") == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/USE_FIXED_SEED","0") == STAT_OK);
    REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    pRootConfig = NULL;
    // run 1
    REQUIRE(runEACirc() == STAT_OK);
    // rename files to be compared
    backupResults();
    REQUIRE(backupFile(FILE_STATE_INITIAL) == STAT_OK);
    REQUIRE(backupFile(FILE_POPULATION_INITIAL) == STAT_OK);
    // prepare run 2
    REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    TiXmlNode* pRootState = NULL;
    REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/USE_FIXED_SEED","1") == STAT_OK);
    REQUIRE(loadXMLFile(pRootState,string(FILE_STATE_INITIAL)+BACKUP_SUFFIX) == STAT_OK);
    string seed = getXMLElementValue(pRootState,"main_seed");
    REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/SEED",seed) == STAT_OK);
    REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    pRootConfig = NULL;
    delete pRootState;
    // run 2
    REQUIRE(runEACirc() == STAT_OK);
    // compare results
    compareResults();
    compareFilesByLine(FILE_STATE_INITIAL,string(FILE_STATE_INITIAL)+BACKUP_SUFFIX);
    compareFilesByLine(FILE_POPULATION_INITIAL,string(FILE_POPULATION_INITIAL)+BACKUP_SUFFIX);
}

TEST_CASE("determinism/load-state","running and running from loaded state") {
    // general preparations
    basicConfiguration::estream();
    TiXmlNode* pRootConfig = NULL;
    // prepare run 1
    REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","0") == STAT_OK);
    REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    pRootConfig = NULL;
    // run 1
    REQUIRE(runEACirc() == STAT_OK);
    // rename files to be compared
    backupResults();
    // prepare run 2
    REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","1") == STAT_OK);
    REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    pRootConfig = NULL;
    REQUIRE(rename(FILE_STATE_INITIAL,FILE_STATE) == STAT_OK);
    REQUIRE(rename(FILE_POPULATION_INITIAL,FILE_POPULATION) == STAT_OK);
    // run 2
    REQUIRE(runEACirc() == STAT_OK);
    // compare results
    compareResults();
}

TEST_CASE("determinism/recommencing","compute 40 generations vs. compute 20+20 generations") {
    // general preparations
    basicConfiguration::estream();
    TiXmlNode* pRootConfig = NULL;
    // prepare run 1
    REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","0") == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/NUM_GENERATIONS","40") == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/SAVE_STATE_FREQ","10") == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/USE_FIXED_SEED","1") == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/SEED","123456789") == STAT_OK);
    REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    pRootConfig = NULL;
    // run 1
    REQUIRE(runEACirc() == STAT_OK);
    // rename files to be compared
    backupResults();
    // prepare run 2
    REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/NUM_GENERATIONS","20") == STAT_OK);
    REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    pRootConfig = NULL;
    // run 2
    REQUIRE(runEACirc() == STAT_OK);
    // prepare run 3
    REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","1") == STAT_OK);
    REQUIRE(setXMLElementValue(pRootConfig,"MAIN/NUM_GENERATIONS","20") == STAT_OK);
    REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    pRootConfig = NULL;
    // run 3
    REQUIRE(runEACirc() == STAT_OK);
    // compare results
    compareResults();
}
