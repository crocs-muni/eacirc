#include "EACglobals.h"
#include "evaluators/IEvaluator.h"
#include "generators/IRndGen.h"
#include "GAPolyCallbacks.h"
#include "PolyDistEval.h"
#include "Term.h"
#include <random>       // std::default_random_engine
#include <algorithm>    // std::move_backward
#include <vector>
#include <array>

void GAPolyCallbacks::initializer(GAGenome& g){
    // Then generate new polynomials using this distribution, minimum is 1 term.
    // (uniform distribution for choosing which variable to include in the polynomial).
    // Define probability distribution on the number of variables in term. Minimum is 1.
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> &genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);
    
    int & numVariables = pGlobals->settings->circuit.sizeInput;
    int & numPolynomials = pGlobals->settings->circuit.sizeOutput;
    int   termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Clear genome.
    for (int i = 0; i < genome.width(); i++) {
        for(int j=0; j < genome.height(); j++){
            genome.gene(i, j);
        }
    }
    
    // How to generate random variables: shuffle a vector.
    std::vector<int> vars;
    for(int i=0; i<numVariables; i++){
        vars.push_back(i);
    }
    
    // Generate each polynomial
    for(int curP = 0; curP < numPolynomials; curP++){
        // Number of terms in polynomial is determined by sampling 
        // geometric distribution / Markov chain. 
        // With certain probability will generating new term stops.
        // Otherwise new ones are generated.
        // To generate polynomial with k terms, probability is: p^{k-1}*p.
        int curTerms;
        for(curTerms = 0; curTerms <  pGlobals->settings->polydist.genomeInitMaxTerms; curTerms++){
            // Generating polynomials with chain
            if (curTerms >= 1 && GAFlipCoin(pGlobals->settings->polydist.genomeInitTermStopProbability)) {
                break;
            }

            // Generate term itself.
            Term t(numVariables);
            
            // Shuffle a variable vector.
            // Variable vector is kept of the same size all the time!
            std::shuffle(vars.begin(), vars.end(), std::default_random_engine(pGlobals->settings->random.seed));
            
            // How many variables should have one term?
            // Same process again -> sample another geometric distribution.
            int curVars = 0;
            for(curVars = 0; curVars < numVariables; curVars++){
                // Generating terms with chain.
                if (curVars >= 1 && GAFlipCoin(pGlobals->settings->polydist.genomeInitTermCountProbability)) {
                    break;
                }
                
                // Add new random variable to the term, remove variable from the var pool.
                t.setBit(vars.at(curVars), 1);
            }
            
            // Dump term to the genome
            t.dumpToGenome(&genome, curP, 1 + curTerms * termSize);
        }
        
        // Set number of terms
        genome.gene(curP, 0, curTerms);
    }
}

float GAPolyCallbacks::evaluator(GAGenome &g) {
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>  &genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);
    
    // reset evaluator state for this individual
    pGlobals->evaluator->resetEvaluator();
    
    // execute circuit & evaluate success for each test vector
    for (int testVector = 0; testVector < pGlobals->settings->testVectors.setSize; testVector++) {
        PolyEval::polyEval(&genome, pGlobals->testVectors.inputs[testVector], pGlobals->testVectors.circuitOutputs[testVector]);
        pGlobals->evaluator->evaluateCircuit(pGlobals->testVectors.circuitOutputs[testVector], pGlobals->testVectors.outputs[testVector]);
    }
    
    // retrieve fitness from evaluator
    return pGlobals->evaluator->getFitness();
}

int GAPolyCallbacks::mutator(GAGenome& g, float probMutation){ 
    int numOfMutations = 0;
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> &genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);
    
    int & numVariables = pGlobals->settings->circuit.sizeInput;
    int & numPolynomials = pGlobals->settings->circuit.sizeOutput;
    unsigned int termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Current mutation strategy: 
    //  - [pick 1 polynomial to mutate ad random]
    //  - pick 1 term from the given polynomial to mutate ad random.
    //  - pick 1 variable from the given term and flip it.
    for(int cPoly=0; cPoly < numPolynomials; cPoly++){
        POLY_GENOME_ITEM_TYPE numTerms = genome.gene(cPoly, 0);
        
        if (GAFlipCoin(probMutation)) {
            // Pick random term
            int randTerm = GARandomInt(0, numTerms-1);

            // Pick random bit
            int randomBit = GARandomInt(0, numVariables-1);

            // Get value of the random bit
            int bitPos = 1 + randTerm*termSize + (randomBit/(8*sizeof(POLY_GENOME_ITEM_TYPE)));
            genome.gene(cPoly, bitPos, genome.gene(cPoly, bitPos) ^ (1ul << (randomBit % (8*sizeof(POLY_GENOME_ITEM_TYPE)))));

            numOfMutations+=1;
        }
        
        // Add a new term to the polynomial?
        if (numTerms < pGlobals->settings->polydist.genomeInitMaxTerms && GAFlipCoin(pGlobals->settings->polydist.mutateAddTermProbability)) {
            // Pick random bit
            int randomBit = GARandomInt(0, numVariables-1);
            
            // New term
            genome.gene(cPoly, 0, numTerms+1);
            
            // Add it - clear storage to have const 1 term by default.
            for(unsigned int i=0; i<termSize; i++){
                genome.gene(cPoly, 1 + numTerms*termSize + i, 0);
            }
            
            int bitPos = 1 + numTerms*termSize + (randomBit/(8*sizeof(POLY_GENOME_ITEM_TYPE)));
            genome.gene(cPoly, bitPos, genome.gene(cPoly, bitPos) ^ (1ul << (randomBit % (8*sizeof(POLY_GENOME_ITEM_TYPE)))));
            
            numTerms+=1;
            numOfMutations+=1;
        }
        
        // Remove a term?
        if (numTerms > 1 && GAFlipCoin(pGlobals->settings->polydist.mutateRemoveTermProbability)){
            // Pick a random term to delete
            unsigned int term2del = GARandomInt(0, numTerms-1);
            
            // Delete term, copy the last term on this place, O(1).
            genome.gene(cPoly, 0, numTerms-1);
            if (term2del < (numTerms-1)){
                // Move the last term to the place of removed.
                for(unsigned int i=0; i<termSize; i++){
                    genome.gene(cPoly, 1 + term2del*termSize + i, genome.gene(cPoly, 1 + (numTerms-1)*termSize + i));
                }
            }
            
            numTerms-=1;
            numOfMutations+=1;
        }
    }
    
    return numOfMutations;
}

int GAPolyCallbacks::crossover(const GAGenome& parent1, const GAGenome& parent2, GAGenome* offspring1, GAGenome* offspring2){
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> * parents[] = {
        (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>*)&parent1, 
        (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>*)&parent2
    };
    
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> * offsprings[] = {
        dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>*>(offspring1), 
        dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>*>(offspring2)
    };
    
    int & numVariables = pGlobals->settings->circuit.sizeInput;
    int & numPolynomials = pGlobals->settings->circuit.sizeOutput;
    unsigned int termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Vectors for generating a random permutation on polynomials.
    std::vector<int> poly1(numPolynomials);
    std::vector<int> poly2(numPolynomials);
    for(int i=0; i<numPolynomials; i++){
        poly1[i] = i;
        poly2[i] = i;
    }
    
    // If want to randomize polynomial selection, shuffle index arrays.
    // TODO: are we generating differen tpermutations here???
    if (pGlobals->settings->polydist.crossoverRandomizePolySelect && numPolynomials > 1){
        std::shuffle(poly1.begin(), poly1.end(), std::default_random_engine(pGlobals->settings->random.seed));
        std::shuffle(poly2.begin(), poly2.end(), std::default_random_engine(pGlobals->settings->random.seed));
    }
    
    // Crossover is very simple here -> uniform selection of the polynomials to the offsprings.
    for(int cPoly = 0; cPoly < numPolynomials; cPoly++){
        // Select next polynomial to the offspring sampling uniform distribution.
        bool pIdx = GAFlipCoin(0.5);
        int pIdxPoly2Pick[] = {poly1[cPoly], poly2[cPoly]};
        
        // Offspring 1.
        // Copy polynomial of the parent0.
        int geneSize = parents[pIdx]->height();
        for(int i=0; i<geneSize; i++){
                offsprings[0]->gene(cPoly, i, parents[pIdx]->gene(pIdxPoly2Pick[pIdx], i)); 
        }
        
        // Offspring 2 - complementary to the offspring 1 w.r.t. polynomial choice.
        geneSize = parents[!pIdx]->height();
        for(int i=0; i<geneSize; i++){
                offsprings[1]->gene(cPoly, i, parents[!pIdx]->gene(pIdxPoly2Pick[!pIdx], i)); 
        }
        
        // If crossover of individual terms is allowed.
        if (GAFlipCoin(pGlobals->settings->polydist.crossoverTermsProbability)) {
            // In this phase enters 2 polynomials, offsprings[0] cPoly, offsprings[1] cPoly.
            POLY_GENOME_ITEM_TYPE numTerms[] = {offsprings[0]->gene(cPoly, 0), offsprings[1]->gene(cPoly, 0)};
            int minTerms = numTerms[0] < numTerms[1] ? 0 : 1;
            
            // Single point crossover on terms, [0, min(t1size, t2size)].
            int crossoverPlace = GARandomInt(0, numTerms[minTerms]-1);
            
            // Do the single point crossover, exchange term size. 
            offsprings[0]->gene(cPoly, 0, numTerms[1]);
            offsprings[1]->gene(cPoly, 0, numTerms[0]);
            
            // Iterate to the maximal number of terms in polynomials.
            // Copy original values / do nothing until crossover point is reached.
            // Then swap genomes.
            for(unsigned int i=crossoverPlace; i<numTerms[!minTerms]; i++){
                // Terms exist for both? If yes, we have to swap terms element-wise.
                if (i < numTerms[minTerms]){
                    // Swapping of the sub-terms, element-wise.
                    for(unsigned int j=0; j<termSize; j++){
                        const int tPos = i*termSize + j;
                        POLY_GENOME_ITEM_TYPE gTmp = offsprings[0]->gene(cPoly, tPos);
                        offsprings[0]->gene(cPoly, tPos, offsprings[1]->gene(cPoly, tPos));
                        offsprings[1]->gene(cPoly, tPos, gTmp);
                    }
                    
                } else {
                    // One term is already finished, no swapping, just copy
                    // the longer tail to just finished term (minTerms)
                    for(unsigned int j=0; j<termSize; j++){
                        const int tPos = i*termSize + j;
                        offsprings[minTerms]->gene(cPoly, tPos, offsprings[!minTerms]->gene(cPoly, tPos));
                    }
                }
            }
        }
    }
    
    // Return number of offsprings.
    return 2;
}

POLY_GENOME_ITEM_TYPE GAPolyCallbacks::changeBit(POLY_GENOME_ITEM_TYPE genomeValue, int width) {
    return genomeValue ^ (1ul << GARandomInt(0, width-1));
}
