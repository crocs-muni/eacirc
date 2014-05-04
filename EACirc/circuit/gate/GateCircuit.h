/* 
 * File:   CircuitRepr.h
 * Author: ph4r05
 *
 * Created on April 29, 2014, 4:29 PM
 */

#ifndef CIRCUITREPR_H
#define	CIRCUITREPR_H

#include "EACglobals.h"
#include "GA1DArrayGenome.h"
#include "circuit/ICircuit.h"
#include "GACallbacks.h"

class GateCircuit : public ICircuit {
public:
    GateCircuit();
    ~GateCircuit();
    string shortDescription();
    
    // Getters for GA callbacks.
    inline GAGenome::Initializer       getInitializer() { return GACallbacks::initializer; }
    inline GAGenome::Evaluator         getEvaluator()   { return GACallbacks::evaluator;   }
    inline GAGenome::Mutator           getMutator()     { return GACallbacks::mutator;     }
    inline GAGenome::Comparator        getComparator()  { return NULL; }
    inline GAGenome::SexualCrossover   getSexualCrossover()  { return GACallbacks::crossover; }
    inline GAGenome::AsexualCrossover  getAsexualCrossover() { return NULL; }
    
    int executeCircuit(GAGenome* pGenome, unsigned char* inputs, unsigned char* outputs) { }

    // Constructs empty genome from settings.
    GAGenome * createGenome(const SETTINGS * settings, bool setCallbacks=false);
    
    // Sets non-null GA callbacks.
    GAGenome * setGACallbacks(GAGenome * g, const SETTINGS * settings);
    
    // Creates a configuration population.
    GAPopulation * createConfigPopulation(const SETTINGS * settings);
    
    bool postProcess(GAGenome &original, GAGenome &prunned);
private:

};

#endif	/* CIRCUITREPR_H */

