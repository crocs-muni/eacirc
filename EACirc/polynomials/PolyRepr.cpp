/* 
 * File:   PolyRepr.cpp
 * Author: ph4r05
 * 
 * Created on April 29, 2014, 4:20 PM
 */

#include "PolyRepr.h"
#include "representation/Repr.h"
#include "PolyIO.h"
#include "GAPolyCallbacks.h"
#include <math.h>

PolyRepr::PolyRepr() {
    initialize();
}

PolyRepr::~PolyRepr() {
    if (io) delete this->io;
    io = NULL;
}

void PolyRepr::initialize(){
    // Fill in IO.
    this->io = new PolyIO();
}

GAGenome::Initializer PolyRepr::getInitializer() {
    return GAPolyCallbacks::initializer;
}

GAGenome::Evaluator PolyRepr::getEvaluator() {
    return GAPolyCallbacks::evaluator;
}

GAGenome::Comparator PolyRepr::getComparator() {
    return NULL;
}

GAGenome::Mutator PolyRepr::getMutator() {
    return GAPolyCallbacks::mutator;
}

GAGenome::SexualCrossover PolyRepr::getSexualCrossover() {
    return GAPolyCallbacks::crossover;
}

GAGenome::AsexualCrossover PolyRepr::getAsexualCrossover() {
    return NULL;
}

GAGenome* PolyRepr::createGenome(const SETTINGS* settings) {
    // Has to compute genome dimensions.
    int numVariables = settings->circuit.sizeInput;
    int numPolynomials = settings->circuit.sizeOutput;
    unsigned int   termElemSize = sizeof(POLY_GENOME_ITEM_TYPE);
    unsigned int   termSize = (int) ceil((double)numVariables / (double)termElemSize);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.    
    
    return new GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>(
            numPolynomials, 
            1 + termSize * settings->polydist.genomeInitMaxTerms,               // number of terms N + N terms.
            this->getEvaluator());
}

GAGenome* PolyRepr::setGACallbacks(GAGenome* g, const SETTINGS* settings) {
    g->initializer(getInitializer());
    g->mutator(getMutator());
    g->crossover(getSexualCrossover());
    return g;
}

GAPopulation* PolyRepr::createConfigPopulation(const SETTINGS* settings) {
    int numVariables = settings->circuit.sizeInput;
    int numPolynomials = settings->circuit.sizeOutput;
    unsigned int   termElemSize = sizeof(POLY_GENOME_ITEM_TYPE);
    unsigned int   termSize = (int) ceil((double)numVariables / (double)termElemSize);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.    
    
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> g(
            numPolynomials, 
            1 + termSize * settings->polydist.genomeInitMaxTerms,               // number of terms N + N terms.
            this->getEvaluator());
    setGACallbacks(&g, settings);
    
    GAPopulation * population = new GAPopulation(g, settings->ga.popupationSize);
    return population;
}

int PolyRepr::postProcess(GAGenome& originalGenome, GAGenome& prunnedGenome) {
    return STAT_NOT_IMPLEMENTED_YET;
}
