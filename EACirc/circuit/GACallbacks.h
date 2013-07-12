#ifndef GACALLBACKS_H
#define GACALLBACKS_H

#include "CircuitGenome.h"

class GACallbacks {
public:
    /** initializes genome
     * @param genome
     */
    static void initializer(GAGenome& genome);

    /** computes individual's fitness
     * @param genome
     * @return fitness value
     */
    static float evaluator(GAGenome& genome);

    /** mutate genome with given probability
     * @param genome
     * @param probability of mutation
     * @return number of mutations performed
     */
    static int mutator(GAGenome& genome, float probMutation);

    /** pair two individuals
     * @param parent1
     * @param parent2
     * @param offspring1
     * @param offspring2
     * @return number of created offsprings
     */
    static int crossover(const GAGenome &parent1, const GAGenome &parent2, GAGenome *offspring1, GAGenome *offspring2);
private:

    /** initialize genome (circuit) into following structure:
     *  1. IN_SELECTOR_LAYER connects inputs to corresponding FNC in the same column (FNC_1_3->IN_3)
     *  2. FUNCTION_LAYER_1 is set to XOR instruction only
     *  3. CONNECTOR_LAYER_i is set to random mask (possibly multiple connectors)
     *  4. FUNCTION_LAYER_i is set to random instruction from range 0..FNC_MAX, respecting allowed instructions in settings
     *     - function argument1 is set to 0
     * @param genome to initialize
     */
    static void initializer_basic(GA1DArrayGenome<GENOM_ITEM_TYPE>& genome);

    /** ???
     * @param genome
     * @return number of mutations performed (currently always 0)
     */
    static int mutator_basic(GA1DArrayGenome<GENOM_ITEM_TYPE>& genome, float probMutation);

    /** make offspings by cutting parents horisontally and exchanging halves
     * crossover point taken randomly
     * @param parent1
     * @param parent2
     * @param offspring1
     * @param offspring2
     * @return number of created offsprings (currently always 1)
     */
    static int crossover_perLayer(const GA1DArrayGenome<GENOM_ITEM_TYPE> &parent1, const GA1DArrayGenome<GENOM_ITEM_TYPE> &parent2,
                                  GA1DArrayGenome<GENOM_ITEM_TYPE> *offspring1, GA1DArrayGenome<GENOM_ITEM_TYPE> *offspring2);

    /** make offspings by cutting parents vertically and exchanging halves
     * crossover point taken randomly
     * @param parent1
     * @param parent2
     * @param offspring1
     * @param offspring2
     * @return number of created offsprings (currently always 1)
     */
    static int crossover_perColumn(const GA1DArrayGenome<GENOM_ITEM_TYPE> &parent1, const GA1DArrayGenome<GENOM_ITEM_TYPE> &parent2,
                                   GA1DArrayGenome<GENOM_ITEM_TYPE> *offspring1, GA1DArrayGenome<GENOM_ITEM_TYPE> *offspring2);
};

#endif // GACALLBACKS_H
