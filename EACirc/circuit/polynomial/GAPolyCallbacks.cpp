#include "EACglobals.h"
#include "evaluators/IEvaluator.h"
#include "generators/IRndGen.h"
#include "GAPolyCallbacks.h"
#include "PolyDistEval.h"
#include "Term.h"
#include "PolynomialCircuit.h"
#include <random>       // std::default_random_engine
#include <algorithm>    // std::move_backward
#include <vector>
#include <array>
#include <iterator>

template<class RandomAccessIterator>
void GAPolyCallbacks::shuffle(RandomAccessIterator first, RandomAccessIterator last) {
    for (auto i=(last-first)-1; i>0; --i) {
        const auto rnd = GARandomInt(0, i);
        swap(first[i], first[rnd]);
    }
}

void GAPolyCallbacks::initializer(GAGenome& g){
    // Then generate new polynomials using this distribution, minimum is 1 term.
    // (uniform distribution for choosing which variable to include in the polynomial).
    // Define probability distribution on the number of variables in term. Minimum is 1.
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> &genome = dynamic_cast<GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&>(g);
    
    const int numVariables =  PolynomialCircuit::getNumVariables();
    const int numPolynomials = PolynomialCircuit::getNumPolynomials();
    const int termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Clear genome.
    for (int i = 0; i < genome.width(); i++) {
        for(int j=0; j < genome.height(); j++){
            genome.gene(i, j, 0);
        }
    }
    
    // How to generate random variables: shuffle a vector.
    // Initialization of variables. Shuffled when needed.
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
        for(curTerms = 0; curTerms <  pGlobals->settings->polyCircuit.genomeInitMaxTerms; curTerms++){
            // Generating polynomials with chain
            if (curTerms >= 1 && GAFlipCoin(pGlobals->settings->polyCircuit.genomeInitTermStopProbability)) {
                break;
            }

            // Generate term itself.
            Term t(numVariables);
            
            // Shuffle a variable vector.
            // Variable vector is kept of the same size all the time!
            shuffle(vars.begin(), vars.end());
            
            // How many variables should have one term?
            // Same process again -> sample another geometric distribution.
            int curVars = 0;
            for(curVars = 0; curVars < numVariables; curVars++){
                // Generating terms with chain.
                if (curVars >= 1 && GAFlipCoin(pGlobals->settings->polyCircuit.genomeInitTermCountProbability)) {
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
    
    const int numVariables = PolynomialCircuit::getNumVariables();
    const int numPolynomials = PolynomialCircuit::getNumPolynomials();
    const unsigned int termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Current mutation strategy: 
    //  - [pick 1 polynomial to mutate ad random]
    //  - pick 1 term from the given polynomial to mutate ad random.
    //  - pick 1 variable from the given term and flip it.
    for(int cPoly=0; cPoly < numPolynomials; cPoly++){
        POLY_GENOME_ITEM_TYPE numTerms = genome.gene(cPoly, 0);
        
        if (GAFlipCoin(probMutation)) {
            // Pick random term
            int randTerm = GARandomInt(0, numTerms-1);

            // Strategy 0 = pick a random bit and flip it.
            // Pros: Quick, easy and clean implementation and semantics. Impl.: O(1).
            // Cons: Provided terms with low weight are more useful, it is more probable
            // this strategy will generate terms with high height thus producing not 
            // useful terms in mutation.
            if (pGlobals->settings->polyCircuit.mutateTermStrategy == MUTATE_TERM_STRATEGY_FLIP){
                // Pick random bit
                int randomBit = GARandomInt(0, numVariables-1);

                // Get value of the random bit
                const int bitPos = Term::getBitPos(randomBit, randTerm, termSize);
                genome.gene(cPoly, bitPos, genome.gene(cPoly, bitPos) ^ (1ul << Term::getBitLoc(randomBit)));

                
            } else if (pGlobals->settings->polyCircuit.mutateTermStrategy == MUTATE_TERM_STRATEGY_ADDREMOVE){
                // ADD/REMOVE strategy: either randomly add a new variable to the term or 
                // randomly remove existing variable from the term. 
                // Pros: Better for hypothesis that shorter terms are more valuable for our purpose (i.e., distinguisher).
                // Cons: Slower implementation, not that clean. Impl.: O(k) if k is size of a term (constant).
                bool addVariable = GAFlipCoin(0.5);
                
                // Build vector of present variables in term, 
                // if we want to add a variable to a term, construct a list of non-set variables.
                // if we want to remove a variable from a term, construct a list of set variables.
                std::vector<int> vars;
                for(int i=0; i<numVariables; i++){
                    const int bitPos = Term::getBitPos(i, randTerm, termSize);
                    const int bitLoc = Term::getBitLoc(i);
                    const bool isVariableInTerm = (genome.gene(cPoly, bitPos) & (1ul << bitLoc)) > 0;
                    
                    // Add to the variable set either if it is present or not.
                    if ((addVariable && !isVariableInTerm) || (!addVariable && isVariableInTerm)){
                        vars.push_back(i);
                    }
                }
                
                // If term is fully saturated (all variables, cannot add),
                // if term is zero, cannot remove. 
                if (vars.size()>0){
                    // Pick one variable at random to either remove or delete.
                    int var2operateOn = vars.at(GARandomInt(0, vars.size()-1));
                    // Toggle specified variable in the term.
                    const int bitPos = Term::getBitPos(var2operateOn, randTerm, termSize);
                    genome.gene(cPoly, bitPos, genome.gene(cPoly, bitPos) ^ (1ul << Term::getBitLoc(var2operateOn)));
                    numOfMutations+=1;
                } 
            } else {
                mainLogger.out(LOGGER_ERROR) << "Unknown mutate term strategy: " << pGlobals->settings->polyCircuit.mutateTermStrategy << endl;
                return 0;
            }
        }
        
        // Add a new term to the polynomial?
        if (numTerms < static_cast<unsigned long>(pGlobals->settings->polyCircuit.genomeInitMaxTerms)){
            for(unsigned int addTermCtr = 0; GAFlipCoin(pGlobals->settings->polyCircuit.mutateAddTermProbability); addTermCtr++){
                // Pick random bit
                int randomBit = GARandomInt(0, numVariables-1);

                // New term
                genome.gene(cPoly, 0, numTerms+1);

                // Add it - clear storage to have const 1 term by default.
                for(unsigned int i=0; i<termSize; i++){
                    genome.gene(cPoly, 1 + numTerms*termSize + i, 0);
                }

                const int bitPos = Term::getBitPos(randomBit, numTerms, termSize);
                genome.gene(cPoly, bitPos, genome.gene(cPoly, bitPos) ^ (1ul << (Term::getBitLoc(randomBit))));

                numTerms+=1;
                numOfMutations+=1;
                
                // Quit, depending on strategy
                if (pGlobals->settings->polyCircuit.mutateAddTermStrategy == MUTATE_ADD_TERM_STRATEGY_ONCE){
                    break;
                }
            }
        }
        
        // Remove a term?
        if (numTerms > 1){
            for(unsigned int rmTermCtr = 0; GAFlipCoin(pGlobals->settings->polyCircuit.mutateRemoveTermProbability); rmTermCtr++){
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
                
                // Quit, depending on strategy
                if (pGlobals->settings->polyCircuit.mutateRemoveTermStrategy == MUTATE_RM_TERM_STRATEGY_ONCE){
                    break;
                }
            }
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
    
    const int numVariables = PolynomialCircuit::getNumVariables();
    const int numPolynomials = PolynomialCircuit::getNumPolynomials();
    const unsigned int termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Vectors for generating a random permutation on polynomials.
    std::vector<int> poly1(numPolynomials);
    std::vector<int> poly2(numPolynomials);
    for(int i=0; i<numPolynomials; i++){
        poly1[i] = i;
        poly2[i] = i;
    }
    
    // If want to randomize polynomial selection, shuffle index arrays.
    if (pGlobals->settings->polyCircuit.crossoverRandomizePolySelect && numPolynomials > 1){
        shuffle(poly1.begin(), poly1.end());
        shuffle(poly2.begin(), poly2.end());
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
        
        // If crossover of individual terms is allowed, perform single crossover
        // on two polynomials, crossing terms.
        if (GAFlipCoin(pGlobals->settings->polyCircuit.crossoverTermsProbability)) {
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
                        const int tPos = 1+i*termSize + j;
                        POLY_GENOME_ITEM_TYPE gTmp = offsprings[0]->gene(cPoly, tPos);
                        offsprings[0]->gene(cPoly, tPos, offsprings[1]->gene(cPoly, tPos));
                        offsprings[1]->gene(cPoly, tPos, gTmp);
                    }
                    
                } else {
                    // One polynomial is already finished, no swapping, just copy
                    // the longer tail to just finished polynomial (minTerms).
                    for(unsigned int j=0; j<termSize; j++){
                        const int tPos = 1+i*termSize + j;
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
