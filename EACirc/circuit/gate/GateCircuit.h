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
    
    inline GAGenome::Initializer       getInitializer();
    inline GAGenome::Evaluator         getEvaluator();
    inline GAGenome::Mutator           getMutator();
    inline GAGenome::Comparator        getComparator();
    inline GAGenome::SexualCrossover   getSexualCrossover();
    inline GAGenome::AsexualCrossover  getAsexualCrossover();

    GAGenome * createGenome(bool setCallbacks = false);
    GAPopulation * createPopulation();
    
    bool postProcess(GAGenome &original, GAGenome &prunned);
    int loadCircuitConfiguration(TiXmlNode* pRoot);
};

#endif	/* GATECIRCUIT_H */

