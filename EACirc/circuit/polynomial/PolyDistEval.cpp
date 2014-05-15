#include "PolyDistEval.h"
#include "set"
#include "Term.h"
#include "PolynomialCircuit.h"

int PolyEval::polyEval(GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome, unsigned char* inputs, unsigned char* outputs){
    // allocate repeatedly used variables
    int numVariables = PolynomialCircuit::getNumVariables();
    int numPolynomials = PolynomialCircuit::getNumPolynomials();
    unsigned int termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Assumption: base type is size of long.
    assert(sizeof(POLY_GENOME_ITEM_TYPE) == sizeof(unsigned long));
    
    // Reset output memory
    memset(outputs, 0, pGlobals->settings->main.circuitSizeOutput);
    
    //
    // Evaluates distinguisher on the given input.
    //
    // Execute each polynomial on the given input.
    for(int cPoly = 0; cPoly < numPolynomials; cPoly++){
        
        // First gene is the length of the current polynomial in terms.
        POLY_GENOME_ITEM_TYPE numTerms = pGenome->gene(cPoly, 0);
        
        // Evaluate polynomial that consists of $numTerms terms.
        // If number of terms in the polynomial is zero, polynomial is evaluated
        // to zero.
        bool polyRes = 0;
        for(POLY_GENOME_ITEM_TYPE cTerm = 0; cTerm < numTerms; cTerm++){
            
            // Each term consists of $termSize term elements.
            // We have assumption term is non-null, thus initialize to 1 by default.
            // Initialization to 0 would cause whole term to be 0 due to way of term evaluation.
            bool ret = 1;
            for(unsigned int i=0; i<termSize; i++){
                POLY_GENOME_ITEM_TYPE cTermEx = pGenome->gene(cPoly, 1 + termSize * cTerm + i);
                ret &= TERM_ITEM_EVAL_GENOME(cTermEx, inputs+i*sizeof(POLY_GENOME_ITEM_TYPE));
            }
            
            // Polynomial is t1 XOR t2 XOR ... XOR t_{numVariables}
            polyRes ^= ret;
        }
        
        // Store result of the polynomial to the output array.
        if (polyRes > 0){
                outputs[cPoly / (8*sizeof(unsigned char))] |= 1ul << (cPoly % (8*sizeof(unsigned char)));
        }
    }
    
    return STAT_OK;
}

int PolyEval::normalize(GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome){
    int numVariables = PolynomialCircuit::getNumVariables();
    int numPolynomials = PolynomialCircuit::getNumPolynomials();
    int termSize = Term::getTermSize(numVariables);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Normalize each polynomial in the distinguisher.
    for(int cPoly = 0; cPoly < numPolynomials; cPoly++){
        
        // First gene is the length of the current polynomial in terms.
        POLY_GENOME_ITEM_TYPE numTerms = pGenome->gene(cPoly, 0);
        
        // Using ordered (tree) set to add all terms to it and then remove...
        std::set<PTerm, PTermComparator> termSet;
        
        // Terms are always nonzero. Removing duplicates has to be done with care
        // since even number of same terms are evaluated to 0, odd number of terms
        // t evaluates to t.
        for(POLY_GENOME_ITEM_TYPE cTerm = 0; cTerm < numTerms; cTerm++){
            PTerm t = new Term(numVariables, pGenome, cPoly, 1 + cTerm * termSize);
            
            // If term is already in the set, do not add it, but remove.
            // TODO: check how equals is performed in the set. using less?
            std::set<PTerm>::iterator existingElem = termSet.find(t);
            if (existingElem == termSet.end()){       // Term is not in the set.
                termSet.insert(t);
            } else if ((*existingElem)->getIgnore()) {
                // Term is already in the set.
                // Equivalent operation (preserving polynomial function)
                // is to remove existing / invalidating it term from the set and ignore current
                // since (a XOR a) = 0.
                // In this if-branch, ignore is set to true, thus no further operation is needed.
            } else {
                // Term is in the set and has ignore=false.
                // Just toggle ignore flag to save operations.
                (*existingElem)->setIgnore(true);
            }
        }
        
        // Dump set to the genome.
        POLY_GENOME_ITEM_TYPE nonNullTerms = 0; 
        std::set<PTerm>::iterator it = termSet.begin();
        for(; it != termSet.end(); it++){
            // If term is set to zero, ignore it.
            if ((*it)->getIgnore()){
                continue;
            }
            
            // Dumps current term to the genome, respecting offset. 
            // One term has $termSize sub-term elements.
            (*it)->dumpToGenome(pGenome, cPoly, 1 + nonNullTerms * termSize);
            nonNullTerms+=1;
        }
        
        // Do not actually resize array since it orthogonal. 
        // Just modify number of terms in the first element.
        // Zero number of terms means polynomial is constantly 0.
        pGenome->gene(cPoly, 0, nonNullTerms);
    }
    
    return STAT_OK;
}
