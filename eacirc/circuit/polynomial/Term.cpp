#include "Term.h"

#include <assert.h>
#include <cmath>
#include <stdexcept>

Term::Term() : size(0), vectorSize(0), term(NULL), ignore(false) { }

Term::~Term() {
    delete this->term;
    this->term = NULL;
}

Term::Term(const Term& other) : size(0), vectorSize(0), term(NULL),  ignore(false) {
    // Initialize internal representation with the size from the source.
    this->setSizes(other.getSize());
    // Copy vectors.
    this->term = new term_t(*(other.term));
}

Term::Term(term_size_t size) : size(0), vectorSize(0), term(NULL),  ignore(false){
    this->initialize(size);
}

Term::Term(term_size_t size, GA2DArrayGenome<unsigned long>* pGenome, const int polyIndex, const int offset)
    : size(0), vectorSize(0), term(NULL), ignore(false) {
    this->initialize(size, pGenome, polyIndex, offset);
}

term_size_t Term::getSize() const {
    return size;
}

Term* Term::setSizes(term_size_t size) {
    this->size = size;
    this->vectorSize = (term_size_t) OWN_CEIL((double) size / (8.0*(double)sizeof(term_elem_t)));
    return this;
}

Term& Term::operator =(const Term& other){
    this->setSizes(other.getSize());

    // Release vector if not null
    if (this->term != NULL){
        delete this->term;
//        this->term->clear(); // from Dusan, to be deleted?
        this->term = NULL;
    }
    // Copy vectors.
    this->term = new term_t(*(other.term));
    // return the existing object
    return *this;
}

Term* Term::initialize(term_size_t size){
    this->setSizes(size);
    return this->initialize();
}

Term* Term::initialize() {
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

Term* Term::initialize(term_size_t size, GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome, const int polyIndex, const int offset) {
    // Assume genome storage is the same to our term store, for code simplicity.
    assert(sizeof(POLY_GENOME_ITEM_TYPE) == sizeof(term_elem_t));

    term_size_t& numVariables = size;
    int termElemSize = sizeof(POLY_GENOME_ITEM_TYPE);
    // Length of one term in terms of POLY_GENOME_ITEM_TYPE.
    int termSize = (int) OWN_CEIL((double)numVariables / (double)termElemSize);
    // Initialize new vector of a specific size inside.
    this->initialize(numVariables);
    // Copy term from the genome.
    for (int i = 0; i < termSize; i++) {
        term->push_back(pGenome->gene(polyIndex, offset + i));
    }
    return this;
}

void Term::dumpToGenome(GA2DArrayGenome<unsigned long>* pGenome, const int polyIndex, const int offset) const {
    // Assume genome storage is the same to our term store, for code simplicity.
    assert(sizeof(POLY_GENOME_ITEM_TYPE) == sizeof(term_elem_t));

    // Iterate over internal term vector and write it to the genome.
    term_t::iterator it1 = term->begin();
    term_t::const_iterator it1End = term->end();
    for (int i = 0; it1 != it1End; it1++, i++){
        pGenome->gene(polyIndex, offset + i, *it1);
    }
}

void Term::setBit(unsigned int bit, bool value){
    if (bit > size){ // TODO? >=
        throw std::out_of_range("illegal bit position");
    }
    const int bitpos = bit / (8*sizeof(term_elem_t));
    if (value) {
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

bool Term::getIgnore() const {
    return this->ignore;
}

Term* Term::setIgnore(bool ignore) {
    this->ignore = ignore;
    return this;
}

bool operator== (const Term &cT1, const Term &cT2) {
    return cT1.compareTo(cT2) == 0;
}

bool operator!= (const Term &cT1, const Term &cT2) {
    return !(cT1 == cT2);
}

int Term::compareTo(const Term& other) const {
    return compareTo(&other);
}

int Term::compareTo(const Term* other) const {
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

bool Term::operator<(const Term& other)   const { return this->compareTo(other) == -1; }
bool Term::operator<=(const Term& other)  const { return this->compareTo(other) !=  1; }
bool Term::operator>(const Term& other)   const { return this->compareTo(other) ==  1; }
bool Term::operator>=(const Term& other)  const { return this->compareTo(other) != -1; }

bool Term::evaluate(const unsigned char* input, term_size_t inputLen) const {
    // We do not allow zero polynomials.
    assert(this->size > 0);
    // For now assume that the size of an internal term element is the same as input.
    assert(sizeof(term_elem_t) >= sizeof(unsigned char));
    // Input length must be at least term size, otherwise we will read invalid memory
    assert(inputLen >= size);

    bool result = 1;
    unsigned int elementIndex = 0;
    for (term_t::iterator element = term->begin() ; element != term->end(); ++element, elementIndex++){
        for (unsigned int byteIndex = 0; byteIndex < sizeof(POLY_GENOME_ITEM_TYPE); byteIndex++){
            // Get byteIndex-th byt of vector
            const unsigned char mask = ((*element)>>(8*byteIndex)) & 0xfful;
            // If mask is null, do not process this input.
            // It may happen the 8*POLY_GENOME_ITEM_TYPE is bigger than
            // number of variables, thus we would read ahead of input array.
            // Term itself must not contain variables out of the range (guarantees
            // that an invalid memory is not read).
            if (mask == 0) continue;
            result &= (*(input+elementIndex*sizeof(POLY_GENOME_ITEM_TYPE)+byteIndex) & mask) == mask;
        }
    }
    return result;
}
