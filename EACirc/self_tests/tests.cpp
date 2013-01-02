#include <iostream>

#include "tests.h"
#include "EACglobals.h"
#include "XMLProcessor.h"
#include "EACirc.h"

void compareFilesByLine(string filename1, string filename2) {
    SCOPED_INFO("################################");
    SCOPED_INFO("Comparing files " << filename1 << " and " << filename2 << ".");
    string line1,line2;
    ifstream file1(filename1);
    ifstream file2(filename2);
    int differCount = 0;
    while ((!file1.eof() || !file2.eof()) && differCount <=5 ) {
        getline(file1,line1);
        getline(file2,line2);
        if (line1 != line2) {
            differCount++;
        }
        CHECK(line1 == line2);
    }
    if (differCount > 5) {
        WARN("Given files (" << filename1 << ", " << filename2 << " differ in more than 5 lines!");
    }
    file1.close();
    file2.close();
}

TEST_CASE("stupid/number equalities", "different numbers are not equal") {
    int number = 5;
    for (int i=1; i<5; i++) {
        CHECK(i !=number);
    }
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
    ofstream config(FILE_CONFIG);
    config << basicConfiguration::estream();
    config.close();
    TiXmlNode* pRootConfig = NULL;
    EACirc* eacirc = NULL;
    // prepare run 1
    //REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
    loadXMLFile(pRootConfig,FILE_CONFIG);
    setXMLElementValue(pRootConfig,"MAIN/LOAD_STATE","0");
    setXMLElementValue(pRootConfig,"RANDOM/USE_FIXED_SEED","1");
    saveXMLFile(pRootConfig,FILE_CONFIG);
    pRootConfig = NULL;
    // run 1
    eacirc = new EACirc(false);
    eacirc->loadConfiguration(FILE_CONFIG);
    eacirc->initializeState();
    eacirc->prepare();
    eacirc->run();
    delete eacirc;
    eacirc = NULL;
    // rename files to be compared
    remove((string(FILE_GALIB_SCORES)+"_2").c_str());
    remove((string(FILE_FITNESS_PROGRESS)+"_2").c_str());
    remove((string(FILE_STATE)+"_2").c_str());
    remove((string(FILE_STATE_INITIAL)+"_2").c_str());
    remove((string(FILE_POPULATION)+"_2").c_str());
    remove((string(FILE_POPULATION_INITIAL)+"_2").c_str());
    rename(FILE_GALIB_SCORES,(string(FILE_GALIB_SCORES)+"_2").c_str());
    rename(FILE_FITNESS_PROGRESS,(string(FILE_FITNESS_PROGRESS)+"_2").c_str());
    rename(FILE_STATE,(string(FILE_STATE)+"_2").c_str());
    rename(FILE_STATE_INITIAL,(string(FILE_STATE_INITIAL)+"_2").c_str());
    rename(FILE_POPULATION,(string(FILE_POPULATION)+"_2").c_str());
    rename(FILE_POPULATION_INITIAL,(string(FILE_POPULATION_INITIAL)+"_2").c_str());
    // prepare run 2
    loadXMLFile(pRootConfig,FILE_CONFIG);
    TiXmlNode* pRootState = NULL;
    //setXMLElementValue(pRootConfig,"RANDOM/USE_FIXED_SEED","1");
    loadXMLFile(pRootState,string(FILE_STATE_INITIAL)+"_2");
    string seed = getXMLElementValue(pRootState,"main_seed");
    //setXMLElementValue(pRootConfig,"RANDOM/SEED",seed);
    saveXMLFile(pRootConfig,FILE_CONFIG);
    pRootConfig = NULL;
    delete pRootState;
    // run 2
    eacirc = new EACirc(false);
    eacirc->loadConfiguration(FILE_CONFIG);
    eacirc->initializeState();
    eacirc->prepare();
    eacirc->run();
    // compare results
    compareFilesByLine(FILE_GALIB_SCORES,string(FILE_GALIB_SCORES)+"_2");
    compareFilesByLine(FILE_FITNESS_PROGRESS,string(FILE_FITNESS_PROGRESS)+"_2");
    compareFilesByLine(FILE_STATE,string(FILE_STATE)+"_2");
    compareFilesByLine(FILE_STATE_INITIAL,string(FILE_STATE_INITIAL)+"_2");
    compareFilesByLine(FILE_POPULATION,string(FILE_POPULATION)+"_2");
    compareFilesByLine(FILE_POPULATION_INITIAL,string(FILE_POPULATION_INITIAL)+"_2");
}
