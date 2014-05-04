/**
 * @file GateCircuit.h
 * @author Martin Ukrop, ph4r05
 */

#ifndef GATECIRCUIT_H
#define	GATECIRCUIT_H

#include "EACglobals.h"
#include "GA1DArrayGenome.h"
#include "circuit/ICircuit.h"
#include "GACallbacks.h"

class GateCircuit : public ICircuit {
public:
    GateCircuit();
    ~GateCircuit();
    string shortDescription();
    
    inline GAGenome::Initializer       getInitializer() { return GACallbacks::initializer; }
    inline GAGenome::Evaluator         getEvaluator()   { return GACallbacks::evaluator;   }
    inline GAGenome::Mutator           getMutator()     { return GACallbacks::mutator;     }
    inline GAGenome::Comparator        getComparator()  { return NULL; }
    inline GAGenome::SexualCrossover   getSexualCrossover()  { return GACallbacks::crossover; }
    inline GAGenome::AsexualCrossover  getAsexualCrossover() { return NULL; }

    GAGenome * createGenome(bool setCallbacks = false);
    GAPopulation * createPopulation();
    
    bool postProcess(GAGenome &original, GAGenome &prunned);
    int loadCircuitConfiguration(TiXmlNode* pRoot);
};

#endif	/* GATECIRCUIT_H */

