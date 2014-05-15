#include "Term.h"

#include <assert.h> 
#include <cmath>
#include <stdexcept>

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
        this->term = new term_t(this->vectorSize, static_cast<term_elem_t>(0));
    } else {
        // Reset vector to zero, fill exact same number of elements as desired.
        term->assign(this->vectorSize, static_cast<term_elem_t>(0));
    }
    
    return this;
}

bool Term::evaluate(const unsigned char * input, term_size_t inputLen) const {
    assert(this->size > 0);
    
    // For now assume that the size of an internal term element
    // is the same as input. 
    assert(sizeof(term_elem_t) >= sizeof(unsigned char));
    
    bool res = 1;
    unsigned int c = 0;
    for (term_t::iterator it = term->begin() ; it != term->end(); ++it, c++){
        for(unsigned int i=0; i<sizeof(POLY_GENOME_ITEM_TYPE); i++){
            const unsigned char mask = ((*it)>>(8*i)) & 0xfful;
            // If mask is null, do not process this input.
            // It may happen the 8*POLY_GENOME_ITEM_TYPE is bigger than 
            // number of variables, thus we would read ahead of input array.
            // Term itself must not contain variables out of the range (guarantees 
            // that an invalid memory is not read).
            if (mask == 0) continue;
            
            res &= (*((input)+i+c*sizeof(POLY_GENOME_ITEM_TYPE)) & mask) == mask;
        }
    }
    
    return res;
}

int Term::compareTo(const Term& other) const {
    return compareTo(&other);
}

int Term::compareTo(const Term * other) const {
    if (this->size < other->getSize()) return  1;
    if (this->size > other->getSize()) return -1;
    
    // Same size here.
    // Compare from the highest variable present.
    term_t::reverse_iterator it1 = this->term->rbegin();
    term_t::reverse_iterator it2 = other->term->rbegin();
    
    term_t::const_reverse_iterator it1End = this->term->rend();
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
    int   termSize = (int) OWN_CEIL((double)numVariables / (double)termElemSize);   // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    
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
    for(int i=0; it1 != it1End; it1++, i++){
        pGenome->gene(polyIdx, offset + i, *it1);
    }
}

void Term::setBit(unsigned int bit, bool value){
    if (bit > size){ 
        throw std::out_of_range("illegal bit position");
    }
    
    const int bitpos = bit / (8*sizeof(term_elem_t));
    if (value){
        term->at(bitpos) = term->at(bitpos) |  (1ul << (bit % (8*sizeof(term_elem_t))));
    } else {
        term->at(bitpos) = term->at(bitpos) & ~(1ul << (bit % (8*sizeof(term_elem_t))));
    }
}

bool Term::getBit(unsigned int bit) const {
    if (bit > size){ 
        throw std::out_of_range("illegal bit position");
    }
    
    return (term->at(bit / (8*sizeof(term_elem_t))) & (bit % (8*sizeof(term_elem_t)))) > 0;
}

void Term::flipBit(unsigned int bit){
    if (bit > size){ 
        throw std::out_of_range("illegal bit position");
    }
    
    term->at(bit / (8*sizeof(term_elem_t))) = term->at(bit / (8*sizeof(term_elem_t))) ^ (1ul << (bit % (8*sizeof(term_elem_t))));
}
    
