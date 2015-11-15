/*
 * File:   PolynomialCircuit.cpp
 * Author: ph4r05
 *
 * Created on April 29, 2014, 4:20 PM
 */

#include "PolynomialCircuit.h"
#include "circuit/ICircuit.h"
#include "PolynomialCircuitIO.h"
#include "GAPolyCallbacks.h"
#include "Term.h"
#include <cmath>
#include "XMLProcessor.h"

PolynomialCircuit::PolynomialCircuit() : ICircuit(CIRCUIT_POLYNOMIAL) { }

PolynomialCircuit::~PolynomialCircuit() { }

string PolynomialCircuit::shortDescription() {
    return "polynomial functions in ANF";
}

GAGenome* PolynomialCircuit::createGenome(bool setCallbacks) {
    // Has to compute genome dimensions.
    int numVariables = PolynomialCircuit::getNumVariables();
    int numPolynomials = PolynomialCircuit::getNumPolynomials();
    // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    unsigned int termSize = Term::getTermSize(numVariables);

    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* g = new GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>(
            numPolynomials,
            // 1 (number of terms) + term size * max number of terms
            1 + termSize * pGlobals->settings->polyCircuit.maxNumTerms,
            this->getEvaluator());
    if (setCallbacks){
        setGACallbacks(g);
    }
    return g;
}

GAPopulation* PolynomialCircuit::createPopulation() {
    GAGenome* g = createGenome(true);
    GAPopulation* population = new GAPopulation(*g, pGlobals->settings->ga.popupationSize);
    delete g;
    return population;
}

bool PolynomialCircuit::postProcess(GAGenome& originalGenome, GAGenome& prunnedGenome) {
    return false;
}

int PolynomialCircuit::loadCircuitConfiguration(TiXmlNode* pRoot) {
    // parsing EACIRC/POLYNOMIAL_CIRCUIT
    pGlobals->settings->polyCircuit.numPolynomials                 = atoi(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/NUM_POLYNOMIALS").c_str());
    pGlobals->settings->polyCircuit.maxNumTerms                    = atoi(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/MAX_NUM_TERMS").c_str());
    pGlobals->settings->polyCircuit.mutateTermStrategy             = atoi(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/MUTATE_TERM_STRATEGY").c_str());
    pGlobals->settings->polyCircuit.genomeInitTermCountProbability = atof(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/TERM_COUNT_PROB").c_str());
    pGlobals->settings->polyCircuit.genomeInitTermStopProbability  = atof(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/TERM_VAR_PROB").c_str());
    pGlobals->settings->polyCircuit.mutateAddTermProbability       = atof(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/ADD_TERM_PROB").c_str());
    pGlobals->settings->polyCircuit.mutateAddTermStrategy          = atoi(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/ADD_TERM_STRATEGY").c_str());
    pGlobals->settings->polyCircuit.mutateRemoveTermProbability    = atof(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/RM_TERM_PROB").c_str());
    pGlobals->settings->polyCircuit.mutateRemoveTermStrategy       = atoi(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/RM_TERM_STRATEGY").c_str());
    pGlobals->settings->polyCircuit.crossoverRandomizePolySelect   = atoi(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/CROSSOVER_RANDOMIZE_POLY").c_str()) ? true : false;
    pGlobals->settings->polyCircuit.crossoverTermsProbability      = atof(getXMLElementValue(pRoot,"POLYNOMIAL_CIRCUIT/CROSSOVER_TERM_PROB").c_str());

    // Sanity checks
    if (pGlobals->settings->polyCircuit.numPolynomials <= 0
            || pGlobals->settings->polyCircuit.numPolynomials > pGlobals->settings->main.circuitSizeOutput*BITS_IN_UCHAR) {
        mainLogger.out(LOGGER_ERROR) << "Number of polynomials <= 0 or > 8 * MAIN/CIRCUIT_SIZE_OUTPUT." << endl;
        return STAT_CONFIG_INCORRECT;
    }
    if (pGlobals->settings->main.evaluatorType != 25
            && pGlobals->settings->polyCircuit.numPolynomials != pGlobals->settings->main.circuitSizeOutput*BITS_IN_UCHAR) {
        mainLogger.out(LOGGER_ERROR) << "Number of polynomials is not 8 * MAIN/CIRCUIT_SIZE_OUTPUT (using evaluator 26 with this polynomials count causes undefined behaviour)." << endl;
        return STAT_CONFIG_INCORRECT;
    }
    if (pGlobals->settings->polyCircuit.genomeInitTermCountProbability < 0 || pGlobals->settings->polyCircuit.genomeInitTermCountProbability > 1) {
        mainLogger.out(LOGGER_ERROR) << "Genome initialization term count probability is out of bounds." << endl;
        return STAT_CONFIG_INCORRECT;
    }
    if (pGlobals->settings->polyCircuit.genomeInitTermStopProbability < 0 || pGlobals->settings->polyCircuit.genomeInitTermStopProbability > 1) {
        mainLogger.out(LOGGER_ERROR) << "Genome initialization term stop probability is out of bounds." << endl;
        return STAT_CONFIG_INCORRECT;
    }
    if (pGlobals->settings->polyCircuit.mutateAddTermProbability < 0 || pGlobals->settings->polyCircuit.mutateAddTermProbability > 1) {
        mainLogger.out(LOGGER_ERROR) << "Mutate add term probability is out of bounds." << endl;
        return STAT_CONFIG_INCORRECT;
    }
    if (pGlobals->settings->polyCircuit.mutateRemoveTermProbability < 0 || pGlobals->settings->polyCircuit.mutateRemoveTermProbability > 1) {
        mainLogger.out(LOGGER_ERROR) << "Mutate remove term probability is out of bounds." << endl;
        return STAT_CONFIG_INCORRECT;
    }
    if (pGlobals->settings->polyCircuit.crossoverTermsProbability < 0 || pGlobals->settings->polyCircuit.crossoverTermsProbability > 1) {
        mainLogger.out(LOGGER_ERROR) << "Crossover terms probability is out of bounds." << endl;
        return STAT_CONFIG_INCORRECT;
    }
    if (pGlobals->settings->ga.replacementSize % 2 == 1) {
        mainLogger.out(LOGGER_ERROR) << "GA/REPLACEMENT_SIZE is odd, polynomials have to use even size of replacement population." << endl;
        return STAT_CONFIG_INCORRECT;
    }

    return STAT_OK;
}
