/* 
 * File:   Repr.h
 * Author: ph4r05
 *
 * Created on April 29, 2014, 3:00 PM
 */

#ifndef REPR_H
#define	REPR_H

#include "EACglobals.h"
#include "GAGenome.h"
#include "GAPopulation.h"
#include "ICircuitIO.h"

class ICircuit {
protected:
    //! circuit type, see EACirc constants
    int m_type;
    //! IO operations for representation, created and deleted automatically
    ICircuitIO* ioCallbackObject;
    
public:
    /** constructor, sets type attribute
     * @param circuit type
     */
    ICircuit(int type);

    /** destructor, deletes ioCallbackObject
     */
    virtual ~ICircuit();
    
    // Getters for GA callbacks.
    virtual GAGenome::Initializer       getInitializer()=0;
    virtual GAGenome::Evaluator         getEvaluator()=0;
    virtual GAGenome::Mutator           getMutator()=0;
    virtual GAGenome::Comparator        getComparator()=0;
    virtual GAGenome::SexualCrossover   getSexualCrossover()=0;
    virtual GAGenome::AsexualCrossover  getAsexualCrossover()=0;
    
    /** execute circuit over given inputs, return outputs
     * @param pGenome       circuit to executeCircuit
     * @param inputs
     * @param outputs
     * @return status
     */
    virtual int executeCircuit(GAGenome* pGenome, unsigned char* inputs, unsigned char* outputs) = 0;

    // Constructs empty genome from settings.
    virtual GAGenome * createGenome(const SETTINGS * settings, bool setCallbacks=false)=0;
    
    // Sets non-null GA callbacks to the genome.
    virtual GAGenome * setGACallbacks(GAGenome * g, const SETTINGS * settings)=0;
    
    // Creates a configuration population.
    virtual GAPopulation * createConfigPopulation(const SETTINGS * settings)=0;
     
    /** individual post-processing, if needed (default does nothing)
     * @param original      genome to post-process
     * @param processed     processed
     * @return did something happen? (i.e. is there valid output in processed?)
     */
    virtual bool postProcess(GAGenome &original, GAGenome &processed);

    /** short textual description of individual representation
      * implementation in representation required!
      * @return description
      */
    virtual string shortDescription() = 0;
    
    /** access io functions
     * @return io callback object
     */
    ICircuitIO * io();

    /** constatnt of active circuit representation
      * @return circuit constant
      */
    int getCircuitType() const;

    /** static function to get correct circuit representation
     * @param circuitType constant
     * @return circuit backend instance or NULL
     */
    static ICircuit* getCircuit(int circuitType);
};

#endif	/* REPR_H */

