/* 
 * File:   Repr.h
 * Author: ph4r05
 *
 * Created on April 29, 2014, 3:00 PM
 */

#ifndef REPR_H
#define	REPR_H

#include "../galib/GAGenome.h"
#include "../galib/GAPopulation.h"
#include "EACglobals.h"
#include "ReprIO.h"

typedef struct GACb_t {
    GAGenome::Initializer       * initializer;
    GAGenome::Evaluator         * evaluator;
    GAGenome::Mutator           * mutator;
    GAGenome::Comparator        * comparator;
    GAGenome::SexualCrossover   * sexualCrossover;
    GAGenome::AsexualCrossover  * asexualCrossover;
} GACb;

class Repr {
protected:
    ReprIO * io;
    //GACb gacb;
    
public:
    Repr();
    virtual ~Repr();
    
    // Internal initializer
    virtual void initialize() { };
    
    // Getters for GA callbacks.
    virtual GAGenome::Initializer       getInitializer()=0;
    virtual GAGenome::Evaluator         getEvaluator()=0;
    virtual GAGenome::Mutator           getMutator()=0;
    virtual GAGenome::Comparator        getComparator()=0;
    virtual GAGenome::SexualCrossover   getSexualCrossover()=0;
    virtual GAGenome::AsexualCrossover  getAsexualCrossover()=0;
    
    // Constructs empty genome from settings.
    virtual GAGenome * createGenome(const SETTINGS * settings)=0;
    
    // Sets non-null GA callbacks to the genome.
    virtual GAGenome * setGACallbacks(GAGenome * g, const SETTINGS * settings)=0;
    
    // Creates a configuration population.
    virtual GAPopulation * createConfigPopulation(const SETTINGS * settings)=0;
    
    // Short description of the representation.
    virtual string shortDescription() { return "Repr"; }
    
    // Individual post-processing.
    virtual int postProcess(GAGenome &originalGenome, GAGenome &prunnedGenome) { return STAT_NOT_IMPLEMENTED_YET; };
    
    // Getter for IO callbacks
    ReprIO * getIOCallbacks() { return this->io; };
    
};

#endif	/* REPR_H */

