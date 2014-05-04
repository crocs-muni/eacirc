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

class Repr {
protected:
    //! circuit type, see EACirc constants
    int m_type;
    // IO operations for representation.
    ReprIO * io;
    
public:
    Repr(int type);
    virtual ~Repr();
    
    // Internal initializer
    virtual void initialize() { }
    
    // Getters for GA callbacks.
    virtual GAGenome::Initializer       getInitializer()=0;
    virtual GAGenome::Evaluator         getEvaluator()=0;
    virtual GAGenome::Mutator           getMutator()=0;
    virtual GAGenome::Comparator        getComparator()=0;
    virtual GAGenome::SexualCrossover   getSexualCrossover()=0;
    virtual GAGenome::AsexualCrossover  getAsexualCrossover()=0;
    
    // Constructs empty genome from settings.
    virtual GAGenome * createGenome(const SETTINGS * settings, bool setCallbacks=false)=0;
    
    // Sets non-null GA callbacks to the genome.
    virtual GAGenome * setGACallbacks(GAGenome * g, const SETTINGS * settings)=0;
    
    // Creates a configuration population.
    virtual GAPopulation * createConfigPopulation(const SETTINGS * settings)=0;
    
    // Short description of the representation.
    virtual string shortDescription() { return "Repr"; }
    
    // Individual post-processing.
    virtual int postProcess(GAGenome &originalGenome, GAGenome &prunnedGenome) { return STAT_NOT_IMPLEMENTED_YET; }
    
    // Getter for IO callbacks
    ReprIO * getIOCallbacks() { return this->io; }

    /** constatnt of active circuit representation
      * @return circuit constant
      */
    int getCircuitType() const;

    /** static function to get correct circuit representation
     * @param circuitType constant
     * @return circuit backend instance or NULL
     */
    static Repr* getCircuit(int circuitType);
};

#endif	/* REPR_H */

