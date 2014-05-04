/* 
 * File:   PolyRepr.h
 * Author: ph4r05
 *
 * Created on April 29, 2014, 4:20 PM
 */

#ifndef POLYREPR_H
#define	POLYREPR_H
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
    
    int executeCircuit(GAGenome* pGenome, unsigned char* inputs, unsigned char* outputs) { }

    // Constructs empty genome from settings.
    virtual GAGenome * createGenome(const SETTINGS * settings, bool setCallbacks=false);
    
    // Sets non-null GA callbacks.
    virtual GAGenome * setGACallbacks(GAGenome * g, const SETTINGS * settings);
    
    // Creates a configuration population.
    virtual GAPopulation * createConfigPopulation(const SETTINGS * settings);
    
    // Individual post-processing.
    virtual bool postProcess(GAGenome &originalGenome, GAGenome &prunnedGenome);
    
private:

};

#endif	/* POLYREPR_H */

