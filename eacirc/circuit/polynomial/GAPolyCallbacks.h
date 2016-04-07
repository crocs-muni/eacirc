#ifndef GAPOLYCALLBACKS_H
#define GAPOLYCALLBACKS_H

#include "EACglobals.h"
#include "PolyCommonFunctions.h"
#include <GAGenome.h>
#include <GA2DArrayGenome.h>

#define SINGLE_TERM_STRATEGY 0

#define MUTATE_TERM_STRATEGY_FLIP 0
#define MUTATE_TERM_STRATEGY_ADDREMOVE 1
#define MUTATE_TERM_STRATEGY_CHANGE 2

#define MUTATE_ADD_TERM_STRATEGY_ONCE 0
#define MUTATE_ADD_TERM_STRATEGY_GEOMETRIC 1

#define MUTATE_RM_TERM_STRATEGY_ONCE 0
#define MUTATE_RM_TERM_STRATEGY_GEOMETRIC 1

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

    /**
     * Random number generator using GAlib (determinism).
     * @param max
     * @return
     */
    inline static int randomGen(int max) { return GARandomInt(0, max-1); }

    /**
     * Shuffling container with RandomAccessIterator.
     * Knuth shuffles in O(n), unbiased algorithm (all permutations equally probable).
     *
     * Uses GA random generator for determinism (if desired).
     *
     * @param first
     * @param last
     */
    template <class RandomAccessIterator>
    static void shuffle(RandomAccessIterator first, RandomAccessIterator last);
};

#endif // GAPOLYCALLBACKS_H
