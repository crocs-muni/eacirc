#include <Catch.h>
#include "CommonFnc.h"
#include "TestConfigurator.h"
#include "XMLProcessor.h"
#include "circuit/gate/GateCommonFunctions.h"
#include "circuit/polynomial/Term.h"
#include "circuit/ICircuit.h"
#include "circuit/polynomial/PolynomialCircuit.h"
#include "circuit/polynomial/PolynomialInterpreter.h"

TEST_CASE("determinism/seed","testing whether run with random seed and second run with the same seed are same") {
    // general preparations
    TestConfigurator configurator;
    configurator.addAllProjects();
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
    configurator.addAllProjects();
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
    configurator.addAllProjects();
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
    configurator.addAllProjects();
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

TEST_CASE("circuit/connector-conversion","relative versus absolute connector masks") {
    pGlobals = new GLOBALS;
    CHECK(relativeToAbsoluteConnectorMask(10,3,6,4) == 20);
    CHECK(relativeToAbsoluteConnectorMask(9,0,6,4) == 18);
    CHECK(absoluteToRelativeConnectorMask(20,3,6,4) == 10);
    CHECK(absoluteToRelativeConnectorMask(18,0,6,4) == 9);
    if (pGlobals != NULL) delete pGlobals;
    pGlobals = NULL;
}

TEST_CASE("polydist/term-eval", "term evaluation") {
    pGlobals = new GLOBALS;
    pGlobals->settings = new SETTINGS();
    pGlobals->settings->main.circuitSizeInput = 2;
    pGlobals->settings->main.circuitSizeOutput = 2;
    pGlobals->settings->polyCircuit.numPolynomials = 16;
    pGlobals->settings->polyCircuit.maxNumTerms = 50;
    int termSize = Term::getTermSize(pGlobals->settings->main.circuitSizeInput*8);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.

    // Polynomial circuit representation.
    PolynomialCircuit circuit;

    // Genome, polynomials will be saved here.
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> * g = (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>*) circuit.createGenome(true);

    // Generate a term, evaluate it on known inputs and observe result.
    Term ** terms = new Term*[16];
    for(int i=0; i<16; i++){
        terms[i] = new Term(16);
    }

    // x_0 * x_4 * x_15
    terms[0]->setBit(0, 1);
    terms[0]->setBit(4, 1);
    terms[0]->setBit(15, 1);

    // x_1
    terms[1]->setBit(1, 1);

    // 1
    terms[2];

    // x_6 * x_14 * x_15
    terms[3]->setBit(6, 1);
    terms[3]->setBit(14, 1);
    terms[3]->setBit(15, 1);

    // poly[0] = term0 XOR term2  = x_0 * x_4 * x_15 + 1
    // poly[1] = term2            = 1
    // poly[2] = term1 XOR term3  = x_1 + x_6 * x_14 * x_15
    // poly[14] = term1 XOR term2 = x_1 + 1
    // poly[15] = term0 XOR term1 = x_0 * x_4 * x_15 + x_6 * x_14 * x_15
    g->gene(0,  0, 2);
    g->gene(1,  0, 1);
    g->gene(2,  0, 2);
    g->gene(14, 0, 2);
    g->gene(15, 0, 2);

    // p1
    terms[0]->dumpToGenome(g, 0, 1+0*termSize);
    terms[2]->dumpToGenome(g, 0, 1+1*termSize);

    // p2
    terms[2]->dumpToGenome(g, 1, 1+0*termSize);

    // p3
    terms[1]->dumpToGenome(g, 2, 1+0*termSize);
    terms[3]->dumpToGenome(g, 2, 1+1*termSize);

    // p4
    terms[1]->dumpToGenome(g, 14, 1+0*termSize);
    terms[2]->dumpToGenome(g, 14, 1+1*termSize);

    // p5
    terms[0]->dumpToGenome(g, 15, 1+0*termSize);
    terms[1]->dumpToGenome(g, 15, 1+1*termSize);

    // Evaluate term on a given inputs
    unsigned char * pInput  = new unsigned char[2];
    unsigned char * pOutput = new unsigned char[2];

    // Macro check.
    // Warning! Endinanness may play role in failing this test.
    pInput[0] = 0xff;             pInput[1] = 0xff;
    CHECK(TERM_ITEM_EVAL_GENOME(0x8011ul, pInput) == 1);

    // 1. evaluation, zero input
    pInput[0] = 0x0;                pInput[1] = 0x0;
    PolynomialInterpreter::executePolynomial(g, pInput, pOutput);
    CHECK(pOutput[0] == 0x3);     CHECK(pOutput[1] == 0x40);

    // 2. evaluation, full input
    pInput[0] = 0xff;             pInput[1] = 0xff;
    PolynomialInterpreter::executePolynomial(g, pInput, pOutput);
    CHECK(pOutput[0] == 0x2);     CHECK(pOutput[1] == 0x0);

    // 3. evaluation, random 1
    pInput[0] = 0x99;             pInput[1] = 0x83;
    PolynomialInterpreter::executePolynomial(g, pInput, pOutput);
    CHECK(pOutput[0] == 0x2);     CHECK(pOutput[1] == 0xc0);

    // 4. evaluation, random 2
    pInput[0] = 0xa4;             pInput[1] = 0x2d;
    PolynomialInterpreter::executePolynomial(g, pInput, pOutput);
    CHECK(pOutput[0] == 0x3);     CHECK(pOutput[1] == 0x40);

    // FREE
    for(int i=0; i<16; i++){
        delete terms[i];
    }
    delete[] terms;
    delete g;
    delete[] pInput;
    delete[] pOutput;
    if (pGlobals != NULL && pGlobals->settings != NULL) {
        delete pGlobals->settings;
        pGlobals->settings = NULL;
    }
    if (pGlobals != NULL) delete pGlobals;
    pGlobals = NULL;
}

TEST_CASE("sanity/floating-point-equality", "Floating point comparison sanity testing") {
    TestConfigurator::floatingPointEqual(0.1,0.1);
    TestConfigurator::floatingPointEqual(1, 10 * 0.1f);
}

// Chi^2 sanity test, reference values taken from WolframAlpha, e.g. 1-CDF[ChiSquareDistrbution[1],3.9]
TEST_CASE("sanity/chi-square-test", "Chi^2 distribution sanity test") {
    TestConfigurator::floatingPointEqual(CommonFnc::chisqr(1, 3.9), 0.0482861);
    TestConfigurator::floatingPointEqual(CommonFnc::chisqr(1, 0.00005), 0.994358);
    TestConfigurator::floatingPointEqual(CommonFnc::chisqr(7, 30), 0.00009495972508134183759741828742325666627324216894472052766221569708930068590721461932866143044174);
    TestConfigurator::floatingPointEqual(CommonFnc::chisqr(7, 3), 0.885002231643150641273266220962340596719670495215043159881425978116885745122995842726570642056612440);
    TestConfigurator::floatingPointEqual(CommonFnc::chisqr(7, 0.5), 0.999446);
    TestConfigurator::floatingPointEqual(CommonFnc::chisqr(255, 300), 0.027727522053904829888992725742535981614753778789868604837431301491523914120345987183486623322619);
    TestConfigurator::floatingPointEqual(CommonFnc::chisqr(255, 250), 0.5766352636499276155128321878596437742748132183501493235263329367246867889129729514290260512191173);
    TestConfigurator::floatingPointEqual(CommonFnc::chisqr(255, 200), 0.995425444541951895268653400299156801014685563033877494690047697443726397137181431060955533820526675);
}

// Kolmogorov-Smirnov sanity test, reference values taken from online calculators (https://home.ubalt.edu/ntsbarsh/business-stat/otherapplets/Uniform.htm)
TEST_CASE("sanity/ks-uniformity-test", "Kolmogorov-Smirnov sanity test") {
    std::vector<double> samples1 = { 0.10, 0.36, 0.22, 0.9, 0.6 };
    std::vector<double> samples2 = { 0.10, 0.10, 0.22, 0.24, 0.42, 0.37, 0.77, 0.99, 0.96, 0.89, 0.85, 0.28, 0.63, 0.09 };
    TestConfigurator::floatingPointEqual(CommonFnc::KS_uniformity_test(samples1), 0.24);
    TestConfigurator::floatingPointEqual(CommonFnc::KS_uniformity_test(samples2), 0.1514286);
}
