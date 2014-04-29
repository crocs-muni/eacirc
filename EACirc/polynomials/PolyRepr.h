/* 
 * File:   PolyRepr.h
 * Author: ph4r05
 *
 * Created on April 29, 2014, 4:20 PM
 */

#ifndef POLYREPR_H
#define	POLYREPR_H
#include "../representation/Repr.h"

class PolyRepr : public Repr {
public:
    // 
    PolyRepr();
    virtual ~PolyRepr();
    
    // Initializer
    virtual void initialize();
    
    // Short description of the representation.
    virtual string shortDescription() { return "PolyRepr"; }
    
    // Getters for GA callbacks.
    virtual GAGenome::Initializer       getInitializer();
    virtual GAGenome::Evaluator         getEvaluator();
    virtual GAGenome::Mutator           getMutator();
    virtual GAGenome::Comparator        getComparator();
    virtual GAGenome::SexualCrossover   getSexualCrossover();
    virtual GAGenome::AsexualCrossover  getAsexualCrossover();
    
    // Constructs empty genome from settings.
    virtual GAGenome * createGenome(const SETTINGS * settings);
    
    // Sets non-null GA callbacks.
    virtual GAGenome * setGACallbacks(GAGenome * g, const SETTINGS * settings);
    
    // Creates a configuration population.
    virtual GAPopulation * createConfigPopulation(const SETTINGS * settings);
    
    // Individual post-processing.
    virtual int postProcess(GAGenome &originalGenome, GAGenome &prunnedGenome);
    
private:

};

#endif	/* POLYREPR_H */

