#ifndef GAPOLYCALLBACKS_H
#define GAPOLYCALLBACKS_H

#include "EACglobals.h"
#include "poly.h"
#include "../galib/GAGenome.h"
#include "../galib/GA2DArrayGenome.h"

class GAPolyCallbacks {
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

    
    /** change one bit somewhere in given width (lower bits)
     * @param genomeValue   value of the genome item to change
     * @param width         number of bits applicable for change
     * @return              changed genome item value
     */
    static POLY_GENOME_ITEM_TYPE changeBit(POLY_GENOME_ITEM_TYPE genomeValue, int width);
};

#endif // GAPOLYCALLBACKS_H
