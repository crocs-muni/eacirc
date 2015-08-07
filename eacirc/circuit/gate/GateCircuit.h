/**
 * @file GateCircuit.h
 * @author Martin Ukrop, ph4r05
 */

#ifndef GATECIRCUIT_H
#define	GATECIRCUIT_H

#include "EACglobals.h"
#include <GA1DArrayGenome.h>
#include "circuit/ICircuit.h"
#include "GAGateCallbacks.h"

class GateCircuit : public ICircuit {
public:
    GateCircuit();

    /** release arrays for interpreter
      */
    virtual ~GateCircuit();

    virtual string shortDescription();

    /** allocate arrays for interpreter
     * @return status
     */
    virtual int initialize();

    virtual GAGenome::Initializer       getInitializer();
    virtual GAGenome::Evaluator         getEvaluator();
    virtual GAGenome::Mutator           getMutator();
    virtual GAGenome::Comparator        getComparator();
    virtual GAGenome::SexualCrossover   getSexualCrossover();
    virtual GAGenome::AsexualCrossover  getAsexualCrossover();

    virtual GAGenome * createGenome(bool setCallbacks = false);
    virtual GAPopulation * createPopulation();

    virtual bool postProcess(GAGenome &original, GAGenome &prunned);
    virtual int loadCircuitConfiguration(TiXmlNode* pRoot);
protected:
    GateCircuit( const int type ) : ICircuit( type ) {}
};

#endif	/* GATECIRCUIT_H */

