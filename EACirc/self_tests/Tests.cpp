#include "Catch.h"
#include "TestConfigurator.h"
#include "XMLProcessor.h"

TEST_CASE("determinism/seed","testing whether run with random seed and second run with the same seed are same") {
    // general preparations
    TestConfigurator configurator;
    while (configurator.nextProject()) {
        configurator.prepareConfiguration();
        TiXmlNode* pRootConfig = NULL;
        // prepare run 1
        REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","0") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/LOAD_INITIAL_POPULATION","0") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/USE_FIXED_SEED","0") == STAT_OK);
        REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        pRootConfig = NULL;
        // run 1
        configurator.runEACirc();
        // rename files to be compared
        configurator.backupResults();
        configurator.backupFile(FILE_STATE_INITIAL);
        configurator.backupFile(FILE_POPULATION_INITIAL);
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
        configurator.runEACirc();
        // compare results
        configurator.compareResults();
        configurator.compareFilesByLine(FILE_STATE_INITIAL,string(FILE_STATE_INITIAL)+BACKUP_SUFFIX);
        configurator.compareFilesByLine(FILE_POPULATION_INITIAL,string(FILE_POPULATION_INITIAL)+BACKUP_SUFFIX);
    }
}

TEST_CASE("determinism/load-state","running and running from loaded state") {
    TestConfigurator configurator;
    while (configurator.nextProject()) {
        // general preparations
        configurator.prepareConfiguration();
        TiXmlNode* pRootConfig = NULL;
        // prepare run 1
        REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","0") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/LOAD_INITIAL_POPULATION","0") == STAT_OK);
        REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        pRootConfig = NULL;
        // run 1
        configurator.runEACirc();
        // rename files to be compared but keep header in fitness progress file
        ifstream fitFileOld(FILE_FITNESS_PROGRESS);
        string header, line;
        getline(fitFileOld,line);
        header += line + "\n";
        getline(fitFileOld,line);
        header += line;
        fitFileOld.close();
        configurator.backupResults();
        ofstream fitFileNew(FILE_FITNESS_PROGRESS);
        fitFileNew << header << endl;
        fitFileNew.close();
        // prepare run 2
        REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","1") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/LOAD_INITIAL_POPULATION","1") == STAT_OK);
        REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        pRootConfig = NULL;
        REQUIRE(rename(FILE_STATE_INITIAL,FILE_STATE) == STAT_OK);
        REQUIRE(rename(FILE_POPULATION_INITIAL,FILE_POPULATION) == STAT_OK);
        // run 2
        configurator.runEACirc();
        // compare results
        configurator.compareResults();
    }
}

TEST_CASE("determinism/recommencing","compute 40 generations vs. compute 20+20 generations") {
    TestConfigurator configurator;
    while (configurator.nextProject()) {
        // general preparations
        configurator.prepareConfiguration();
        TiXmlNode* pRootConfig = NULL;
        // prepare run 1
        REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","0") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/LOAD_INITIAL_POPULATION","0") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/NUM_GENERATIONS","40") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/SAVE_STATE_FREQ","10") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/USE_FIXED_SEED","1") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/SEED","123456789") == STAT_OK);
        REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        pRootConfig = NULL;
        // run 1
        configurator.runEACirc();
        // rename files to be compared
        configurator.backupResults();
        // prepare run 2
        REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/NUM_GENERATIONS","20") == STAT_OK);
        REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        pRootConfig = NULL;
        // run 2
        configurator.runEACirc();
        // prepare run 3
        REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","1") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/LOAD_INITIAL_POPULATION","1") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/NUM_GENERATIONS","20") == STAT_OK);
        REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        pRootConfig = NULL;
        // run 3
        configurator.runEACirc();
        // compare results
        configurator.compareResults();
    }
}

TEST_CASE("determinism/tv-pregeneration","general run vs. run with the same pre-generated test vectors") {
    TestConfigurator configurator;
    while (configurator.nextProject()) {
        // general preparations
        configurator.prepareConfiguration();
        TiXmlNode* pRootConfig = NULL;
        // prepare run 1
        REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/RECOMMENCE_COMPUTATION","0") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/LOAD_INITIAL_POPULATION","0") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/NUM_GENERATIONS","40") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"MAIN/SAVE_STATE_FREQ","10") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/USE_FIXED_SEED","1") == STAT_OK);
        REQUIRE(setXMLElementValue(pRootConfig,"RANDOM/SEED","123456789") == STAT_OK);
        REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        pRootConfig = NULL;
        // run 1
        configurator.runEACirc();
        // rename files to be compared
        configurator.backupResults();
        // prepare run 2
        REQUIRE(loadXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        REQUIRE(saveXMLFile(pRootConfig,FILE_CONFIG) == STAT_OK);
        pRootConfig = NULL;
        // run 2
        configurator.runEACirc();
        // compare results
        configurator.compareResults();
    }
}
