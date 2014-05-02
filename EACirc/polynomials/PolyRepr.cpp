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
#include "Term.h"
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

GAGenome* PolyRepr::createGenome(const SETTINGS* settings, bool setCallbacks) {
    // Has to compute genome dimensions.
    int numVariables = settings->polydist.numVariables;
    int numPolynomials = settings->polydist.numPolynomials;
    unsigned int   termSize = Term::getTermSize(numVariables); // Length of one term in terms of POLY_GENOME_ITEM_TYPE.    
    
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> * g = new GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>(
            numPolynomials, 
            1 + termSize * settings->polydist.genomeInitMaxTerms,               // number of terms N + N terms.
            this->getEvaluator());
    
    if (setCallbacks){
        setGACallbacks(g, settings);
    }
    
    return g;
}

GAGenome* PolyRepr::setGACallbacks(GAGenome* g, const SETTINGS* settings) {
    g->initializer(getInitializer());
    g->evaluator(getEvaluator());
    g->mutator(getMutator());
    g->crossover(getSexualCrossover());
    return g;
}

GAPopulation* PolyRepr::createConfigPopulation(const SETTINGS* settings) {
    int numVariables = settings->polydist.numVariables;
    int numPolynomials = settings->polydist.numPolynomials;
    unsigned int   termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.    
    
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
