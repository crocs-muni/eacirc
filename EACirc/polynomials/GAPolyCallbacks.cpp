#include "EACglobals.h"
#include "evaluators/IEvaluator.h"
#include "generators/IRndGen.h"
#include "GAPolyCallbacks.h"
#include "PolyDistEval.h"
#include "Term.h"
#include <random>       // std::default_random_engine
#include <algorithm>    // std::move_backward
#include <vector>

void GAPolyCallbacks::initializer(GAGenome& g){
    // Then generate new polynomials using this distribution, minimum is 1 term.
    // (uniform distribution for choosing which variable to include in the polynomial).
    // Define probability distribution on the number of variables in term. Minimum is 1.
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> &genome = (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&) g;
    
    int & numVariables = pGlobals->settings->circuit.sizeInput;
    int & numPolynomials = pGlobals->settings->circuit.sizeOutput;
    int   termElemSize = sizeof(POLY_GENOME_ITEM_TYPE);
    int   termSize = (int) ceil((double)numVariables / (double)termElemSize);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
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
            if (curTerms > 1 && GAFlipCoin(pGlobals->settings->polydist.genomeInitTermStopProbability)) {
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
                if (curVars > 1 && GAFlipCoin(pGlobals->settings->polydist.genomeInitTermCountProbability)) {
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
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>  &genome = (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&) g;
    
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
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> &genome = (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>&) g;
    
    int & numVariables = pGlobals->settings->circuit.sizeInput;
    int & numPolynomials = pGlobals->settings->circuit.sizeOutput;
    int   termElemSize = sizeof(POLY_GENOME_ITEM_TYPE);
    int   termSize = (int) ceil((double)numVariables / (double)termElemSize);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Current mutation strategy: 
    //  - [pick 1 polynomial to mutate ad random]
    //  - pick 1 term from the given polynomial to mutate ad random.
    //  - pick 1 variable from the given term and flip it.
    for(int cPoly=0; cPoly < numPolynomials; cPoly++){
        if (!GAFlipCoin(probMutation)) continue;
        
        // Pick random term
        POLY_GENOME_ITEM_TYPE termSize = genome.gene(cPoly, 0);
        int randTerm = GARandomInt(0, termSize-1);
        
        // Pick random bit
        int randomBit = GARandomInt(0, numVariables-1);
        
        // Get value of the random bit
        int bitPos = 1+randTerm*termSize + (randomBit/sizeof(POLY_GENOME_ITEM_TYPE));
        genome.gene(cPoly, bitPos, genome.gene(cPoly, bitPos) ^ (1 << (randomBit % sizeof(POLY_GENOME_ITEM_TYPE))));
        
        numOfMutations+=1;
    }
    
    return numOfMutations;
}

int GAPolyCallbacks::crossover(const GAGenome& parent1, const GAGenome& parent2, GAGenome* offspring1, GAGenome* offspring2){
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> * parents[] = {
        (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>*) &parent1, 
        (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>*) &parent2
    };
    
    GA2DArrayGenome<POLY_GENOME_ITEM_TYPE> * offsprings[] = {
        (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>*) offspring1, 
        (GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>*) offspring2
    };
    
    // Crossover is very simple here -> uniform selection of the polynomials to the offsprings.
    int & numPolynomials = pGlobals->settings->circuit.sizeOutput;
    for(int cPoly = 0; cPoly < numPolynomials; cPoly++){
        // Select next polynomial to the offspring sampling uniform distribution.
        bool pIdx = GAFlipCoin(0.5);
        
        // Offspring 1.
        // Copy polynomial of the parent0.
        int geneSize = parents[pIdx]->width();
        for(int i=0; i<geneSize; i++){
                offsprings[0]->gene(cPoly, i, parents[pIdx]->gene(cPoly, i)); 
        }
        
        // Offspring 2 - complementary to the offspring 1 w.r.t. polynomial choice.
        geneSize = parents[!pIdx]->width();
        for(int i=0; i<geneSize; i++){
                offsprings[1]->gene(cPoly, i, parents[!pIdx]->gene(cPoly, i)); 
        }
    }
    
    // Return number of offsprings.
    return 2;
}

POLY_GENOME_ITEM_TYPE GAPolyCallbacks::changeBit(POLY_GENOME_ITEM_TYPE genomeValue, int width) {
    return genomeValue ^ (1 << GARandomInt(0, width-1));
}
