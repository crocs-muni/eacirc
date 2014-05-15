/**
 * @file ICircuit.h
 * @author Martin Ukrop, ph4r05
 */

#ifndef ICIRCUIT_H
#define	ICIRCUIT_H

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

    /** general initialization, empty by default
     * @return status
     */
    virtual int initialize();
    
    /** short textual description of individual representation
      * implementation in representation required!
      * @return description
      */
    virtual string shortDescription() = 0;

    // getters for GA callbacks, return NULL if not available
    virtual GAGenome::Initializer       getInitializer()=0;
    virtual GAGenome::Evaluator         getEvaluator()=0;
    virtual GAGenome::Mutator           getMutator()=0;
    virtual GAGenome::Comparator        getComparator()=0;
    virtual GAGenome::SexualCrossover   getSexualCrossover()=0;
    virtual GAGenome::AsexualCrossover  getAsexualCrossover()=0;

    /** construct new genome according to global settings
     * @param setCallbacks  should GAcallbacks be set in the genome object?
     * @return genome
     */
    virtual GAGenome * createGenome(bool setCallbacks = false) = 0;

    /** construct new population according to global settings
     * - GA callbacks are automatically set
     * @return population
     */
    virtual GAPopulation * createPopulation() = 0;
    
    /** set callbacks to given genome
     * - default sets all callbacks (NULL = not available)
     * - sexual crossover is prefered over asexual
     * @param g     genome
     */
    virtual void setGACallbacks(GAGenome * g);
    
     
    /** individual post-processing, if needed (default does nothing)
     * @param original      genome to post-process
     * @param processed     processed
     * @return did something happen? (i.e. is there valid output in processed?)
     */
    virtual bool postProcess(GAGenome &original, GAGenome &processed);
    
    /** load circuit representation-specific configuration
      * default implementation: load no configuration
      * @param pRoot    parsed XML tree with configuration (root=EACIRC)
      * @return status
      */
    virtual int loadCircuitConfiguration(TiXmlNode* pRoot);

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

#endif	/* ICIRCUIT_H */
