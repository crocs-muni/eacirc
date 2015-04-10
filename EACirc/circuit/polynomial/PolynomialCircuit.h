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
    virtual string shortDescription();

    // Getters for GA callbacks.
    virtual inline GAGenome::Initializer       getInitializer() { return GAPolyCallbacks::initializer; }
    virtual inline GAGenome::Evaluator         getEvaluator()   { return GAPolyCallbacks::evaluator;   }
    virtual inline GAGenome::Mutator           getMutator()     { return GAPolyCallbacks::mutator;     }
    virtual inline GAGenome::Comparator        getComparator()  { return NULL; }
    virtual inline GAGenome::SexualCrossover   getSexualCrossover()  { return GAPolyCallbacks::crossover; }
    virtual inline GAGenome::AsexualCrossover  getAsexualCrossover() { return NULL; }

    /** Constructs empty genome from settings.
     * @param setCallbacks      should all GA callbacks be set?
     * @return                  newly allocated 2D genome
     */
    virtual GAGenome* createGenome(bool setCallbacks = false);

    /** Allocates a new population according to global settings.
     * - callbacks are set
     * @return      newly allocated population
     */
    virtual GAPopulation* createPopulation();

    // Individual post-processing.
    virtual bool postProcess(GAGenome &originalGenome, GAGenome &prunnedGenome);

    /** Process configuration sub-tree for polynomial back-end.
     * @param pRoot     parsed XML tree with configuration (root=EACIRC)
     * @return          status
     */
    int loadCircuitConfiguration(TiXmlNode* pRoot);

    /** Obtains number of variables from the global configuration.
     * => i.e. number of bits for input
     * @return
     */
    static inline int getNumVariables() { return pGlobals->settings->main.circuitSizeInput*8; }

    /** Obtains number of polynomials from the global configuration.
     * => i.e. number of independent polynomials (output bits)
     * @return
     */
    static inline int getNumPolynomials() { return pGlobals->settings->polyCircuit.numPolynomials; }

private:

};

#endif	/* POLYNOMIALCIRCUIT_H */

