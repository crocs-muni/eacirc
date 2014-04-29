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

CircuitRepr::CircuitRepr() {
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

GAGenome::Initializer CircuitRepr::getInitializer() {
    return GACallbacks::initializer;
}

GAGenome::Evaluator CircuitRepr::getEvaluator() {
    return GACallbacks::evaluator;
}

GAGenome::Comparator CircuitRepr::getComparator() {
    return NULL;
}

GAGenome::Mutator CircuitRepr::getMutator() {
    return GACallbacks::mutator;
}

GAGenome::SexualCrossover CircuitRepr::getSexualCrossover() {
    return GACallbacks::crossover;
}

GAGenome::AsexualCrossover CircuitRepr::getAsexualCrossover() {
    return NULL;
}

GAGenome* CircuitRepr::createGenome(const SETTINGS* settings) {
    return new GA1DArrayGenome<GENOME_ITEM_TYPE>(settings->circuit.genomeSize, getEvaluator());
}

GAGenome* CircuitRepr::setGACallbacks(GAGenome* g, const SETTINGS* settings) {
    g->initializer(getInitializer());
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

