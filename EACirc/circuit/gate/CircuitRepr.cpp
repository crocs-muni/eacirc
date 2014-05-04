/* 
 * File:   CircuitRepr.cpp
 * Author: ph4r05
 * 
 * Created on April 29, 2014, 4:29 PM
 */

#include "CircuitRepr.h"
#include "CircuitIO.h"
#include "CircuitCommonFunctions.h"
#include "GACallbacks.h"
#include "CircuitInterpreter.h"
#include "GAPopulation.h"

CircuitRepr::CircuitRepr() : Repr(CIRCUIT_GATE) {
    initialize();
}

CircuitRepr::~CircuitRepr() {
    if (io) delete this->io;
    io = NULL;
}

void CircuitRepr::initialize(){
    // Fill in IO.
    this->io = new CircuitIO();
}

GAGenome* CircuitRepr::createGenome(const SETTINGS* settings, bool setCallbacks) {
    GA1DArrayGenome<GENOME_ITEM_TYPE> *g = new GA1DArrayGenome<GENOME_ITEM_TYPE>(settings->circuit.genomeSize, getEvaluator());
    if (setCallbacks){
        setGACallbacks(g, settings);
    }
    
    return g;
}

GAGenome* CircuitRepr::setGACallbacks(GAGenome* g, const SETTINGS* settings) {
    g->initializer(getInitializer());
    g->evaluator(getEvaluator());
    g->mutator(getMutator());
    g->crossover(getSexualCrossover());
    return g;
}

GAPopulation* CircuitRepr::createConfigPopulation(const SETTINGS* settings) {
    GA1DArrayGenome<GENOME_ITEM_TYPE> g(settings->circuit.genomeSize, getEvaluator());
    setGACallbacks(&g, settings);
    
    GAPopulation * population = new GAPopulation(g, settings->ga.popupationSize);
    return population;
}

int CircuitRepr::postProcess(GAGenome& originalGenome, GAGenome& prunnedGenome) {
    return CircuitInterpreter::pruneCircuit(originalGenome, prunnedGenome);
}

