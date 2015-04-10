/**
 * @file GateCircuit.h
 * @author Martin Ukrop, ph4r05
 */

#ifndef GATECIRCUIT_H
#define	GATECIRCUIT_H

#include "EACglobals.h"
#include "GA1DArrayGenome.h"
#include "circuit/ICircuit.h"
#include "GAGateCallbacks.h"

class GateCircuit : public ICircuit {
public:
    GateCircuit();

    /** release arrays for interpreter
      */
    ~GateCircuit();

    string shortDescription();

    /** allocate arrays for interpreter
     * @return status
     */
    int initialize();

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

