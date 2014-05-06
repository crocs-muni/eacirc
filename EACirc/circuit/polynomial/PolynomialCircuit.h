/* 
 * File:   PolynomialCircuit.h
 * Author: ph4r05
 *
 * Created on April 29, 2014, 4:20 PM
 */

#ifndef POLYNOMIALCIRCUIT_H
#define	POLYNOMIALCIRCUIT_H
#include "circuit/ICircuit.h"
#include "GAPolyCallbacks.h"

class PolynomialCircuit : public ICircuit {
public:
    // 
    PolynomialCircuit();
    virtual ~PolynomialCircuit();
    
    // Short description of the representation.
    virtual string shortDescription() { return "PolyRepr"; }
    
    // Getters for GA callbacks.
    virtual inline GAGenome::Initializer       getInitializer() { return GAPolyCallbacks::initializer; }
    virtual inline GAGenome::Evaluator         getEvaluator()   { return GAPolyCallbacks::evaluator;   }
    virtual inline GAGenome::Mutator           getMutator()     { return GAPolyCallbacks::mutator;     }
    virtual inline GAGenome::Comparator        getComparator()  { return NULL; }
    virtual inline GAGenome::SexualCrossover   getSexualCrossover()  { return GAPolyCallbacks::crossover; }
    virtual inline GAGenome::AsexualCrossover  getAsexualCrossover() { return NULL; }

    // Constructs empty genome from settings.
    virtual GAGenome * createGenome(bool setCallbacks = false);
    
    // Creates a configuration population.
    virtual GAPopulation * createPopulation();
    
    // Individual post-processing.
    virtual bool postProcess(GAGenome &originalGenome, GAGenome &prunnedGenome);

    int loadCircuitConfiguration(TiXmlNode* pRoot);
    
    /** Obtains number of variables from the global configuration.
     * @return 
     */
    inline static int getNumVariables() { return pGlobals->settings->main.circuitSizeInput*8; }

    /** Obtains number of polynomials from the global configuration.
     * @return 
     */
    inline static int getNumPolynomials() { return pGlobals->settings->polyCircuit.numPolynomials; }

private:

};

#endif	/* POLYNOMIALCIRCUIT_H */

