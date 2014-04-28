#include "Term.h"

#include <assert.h> 
#include <cmath>

Term::Term(const Term& cT){
    // Initialize internal representation with the size from the source.
    this->setSize(cT.getSize());
    
    // Copy vectors.
    this->term = new term_t(*(cT.term));
}

Term::Term(term_size_t size){
    this->initialize(size);
}

Term::Term(term_size_t size, GA2DArrayGenome<unsigned long>* pGenome, const int polyIdx, const int offset){
    this->initialize(size, pGenome, polyIdx, offset);
}

Term& Term::operator =(const Term& other){
    this->setSize(other.getSize());
    
    // Release vector if not null
    if (this->term != NULL){
        this->term->clear();
        this->term = NULL;
    }
    
    // Copy vectors.
    this->term = new term_t(*(other.term));
 
    // return the existing object
    return *this;
}

Term * Term::initialize(term_size_t size){
    this->setSize(size);
    return this->initialize();
}

Term * Term::initialize() {
    
    // Null terms are not allowed, it has to be initialized prior this call.
    assert(this->size > 0);
    
    // Clear & reserve space in vector.
    if (this->term == NULL){
        this->term = new term_t(this->vectorSize);
    } else {
        this->term->clear();
        this->term->reserve(this->vectorSize);
    }
    
    return this;
}

bool Term::evaluate(const unsigned char * input, term_size_t inputLen) const {
    assert(this->size > 0);
    
    // For now assume that the size of an internal term element
    // is the same as input. 
    assert(sizeof(term_elem_t) == sizeof(char));
    
    bool res = 1;
    unsigned int c = 0;
    for (term_t::iterator it = term->begin() ; it != term->end(); ++it, c++){
        res &= (input[c] & (*it)) == (*it);
    }
    
    return res;
}

int Term::compareTo(const Term& other) const {
    return compareTo(&other);
}

int Term::compareTo(const Term * other) const {
    if (size < other->getSize()) return  1;
    if (size > other->getSize()) return -1;
    
    // Same size here.
    // Compare from the highest variable present.
    term_t::reverse_iterator it1 = term->rbegin();
    term_t::reverse_iterator it2 = other->term->rbegin();
    
    term_t::const_reverse_iterator it1End = term->rend();
    term_t::const_reverse_iterator it2End = other->term->rend();
    for(;it1 != it1End && it2 != it2End; it1++, it2++){
        if ((*it1) < (*it2)) return  1;
        if ((*it1) > (*it2)) return -1;
    }
    
    return 0;
}

Term * Term::initialize(term_size_t size, GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome, const int polyIdx, const int offset){
    // Assume genome storage is the same to our term store, for code simplicity.
    assert(sizeof(POLY_GENOME_ITEM_TYPE) == sizeof(term_elem_t));
    
    term_size_t & numVariables = size;
    int   termElemSize = sizeof(POLY_GENOME_ITEM_TYPE);
    int   termSize = (int) ceil((double)numVariables / (double)termElemSize);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
    // Initialize new vector of a specific size inside.
    this->initialize(numVariables);
    
    // Copy term from the genome.
    for(int i=0; i<termSize; i++){
        term->push_back(pGenome->gene(polyIdx, offset + i));
    }
    
    return this;
}

void Term::dumpToGenome(GA2DArrayGenome<unsigned long>* pGenome, const int polyIdx, const int offset) const {
    // Assume genome storage is the same to our term store, for code simplicity.
    assert(sizeof(POLY_GENOME_ITEM_TYPE) == sizeof(term_elem_t));
    
    // Iterate over internal term vector and write it to the genome.
    term_t::iterator it1 = term->begin();
    term_t::const_iterator it1End = term->end();
    for(int i=0;it1 != it1End; it1++, i++){
        pGenome->gene(polyIdx, offset + i, *it1);
    }
}

bool Term::setBit(unsigned int bit, bool value){
    if (bit > size){ 
        return false; // TODO: throw exception
    }
    
    if (value){
        term->at(bit / sizeof(term_elem_t)) |  (1 << (bit % sizeof(term_elem_t)));
    } else {
        term->at(bit / sizeof(term_elem_t)) & ~(1 << (bit % sizeof(term_elem_t)));
    }
    return true;
}

bool Term::getBit(unsigned int bit) const {
    if (bit > size){ 
        return false; // TODO: throw exception
    }
    
    return (term->at(bit / sizeof(term_elem_t)) & (bit % sizeof(term_elem_t))) > 0;
}

bool Term::flipBit(unsigned int bit){
    if (bit > size){ 
        return false; // TODO: throw exception
    }
    
    term->at(bit / sizeof(term_elem_t)) = term->at(bit / sizeof(term_elem_t)) ^ (1 << (bit % sizeof(term_elem_t)));
    return true;
}
    
