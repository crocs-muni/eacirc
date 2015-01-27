#ifndef _TERM_H
#define _TERM_H

#include "PolyCommonFunctions.h"
#include <assert.h>
#include <vector>
#include <cmath>
#include <stdexcept>

// Basic element of the term array/vector.
typedef POLY_GENOME_ITEM_TYPE term_elem_t;

// Number of the variables in a term type.
typedef unsigned long term_size_t;

// Typedef for the vector inside term
typedef std::vector<term_elem_t> term_t;

/**
 * GF2[x1, ..., xn]
 *
 * One term in a polynomial.
 * Consists of an array of term_elem_t.
 *
 * Note: Null term (all term_elem_t are zero) will
 * be evaluated to 1.
 *
 * It is not possible to express a zero term (not needed in
 * the polynomial representation of a function).
 *
 * Term is stored in the internal vector in this way:
 * x_0  .. x_63  goes to the vector[0]
 * x_64 .. x_128 goes to the vector[1] ...
 *
 */
class Term {
  protected:
    /**
     * Number of input variables.
     * i.e., bit input size.
     */
    term_size_t size;

    /**
     * Size of the vector derived from the size.
     * ceil(size / 8*sizeof(term_elem_t))
     */
    term_size_t vectorSize;

    /**
     * Internal term representation using a vector.
     */
    term_t* term;

    /**
     * Helper attribute - whether to ignore this term or not.
     * Used in transforming to ANF.
     * If true, this term should not appear in the ANF.
     *
     * @param cT
     */
    bool ignore;

  public:
    /**
     * Default constructor.
     * Construct an empty term, internal vector is not initialized.
     */
    Term();

    /**
     * Destructor.
     */
    ~Term();

    /**
     * Copy constructor.
     * Copy sizes and internal vector.
     * @param other     term to copy
     */
    Term(const Term &other);

    /**
     * Instantiate using size, internal vector is reinitialized and cleaned.
     * @param size      desired size
     */
    Term(term_size_t size);

    /**
     * Instantiate using genome.
     */
    Term(term_size_t size, GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome, const int polyIdx, const int offset);

    /**
     * Getter for the size.
     * @return size
     */
    term_size_t getSize(void) const;

    /**
     * Returns number of term building blocks if term has provided bit size.
     * @param bitSize
     * @return
     */
    static inline unsigned int getTermSize(unsigned int bitSize) {
        return OWN_CEIL((double) bitSize / (8.0*(double)sizeof(term_elem_t)));
    }

    /**
     * Returns number of term building blocks if term has provided bit size, provided storage type.
     * @param bitSize
     * @param typeSize
     * @return
     */
    static inline unsigned int getTermSize(unsigned int bitSize, unsigned int typeSize) {
        return OWN_CEIL((double) bitSize / (8.0*(double)typeSize));
    }

    /**
     * Setter only for the size and vectorSize. Performs no initialization.
     */
    Term* setSizes(term_size_t size);

    /**
     * Term initializer, set size has to be called before.
     */
    Term* initialize();

    /**
     * Term initializer, resizes the container if necessary.
     * @param size      new term size
     * @return self
     */
    Term* initialize(term_size_t size);

    /** Initialize term from the genome.
     * @param pGenome           genome for reading
     * @param polyIdx           1. D index (which polynomial to use)
     * @param offset            2. D offset (where to start reading)
     * @return self
     */
    Term* initialize(term_size_t size, GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome, const int polyIdx, const int offset);

    /** Dumps term to the genome.
     * Writes vector[0] .. vector[n] to the genome (thus ordering is x_0 ... x_size).
     * @param pGenome           genome to write to
     * @param polyIdx           D index (which polynomial to use)
     * @param offset            D offset (where to start writing)
     */
    void dumpToGenome(GA2DArrayGenome<POLY_GENOME_ITEM_TYPE>* pGenome, const int polyIdx, const int offset) const;

    /** Sets particular bit in the term.
     * @throws          out_of_range exception, if bit position is too big
     * @param bit       bt position to write
     * @param value     desired value
     */
    void setBit(unsigned int bit, bool value);

    /** Gets particular bit value in the term.
     * @throws          out_of_range exception, if bit position is too big
     * @param bit       bit position to read
     * @return          bit value
     */
    bool getBit(unsigned int bit) const;

    /** Flips particular bit value in the term.
     * @throws          out_of_range exception, if bit position is too big
     * @param bit       bit position to read
     */
    void flipBit(unsigned int bit);

    /** Get ignore flag.
     * @return          ignore flag
     */
    bool getIgnore() const;

    /** Set ignore flag.
     * @param ignore    flag to set
     * @return          self
     */
    Term* setIgnore(bool ignore);

    // Comparators for sorting
    //  1: I'm smaller than the other
    // -1: I'm bigger than the other
    int compareTo(const Term& other) const;
    int compareTo(const Term* other) const;

    // Assignment operator
    Term& operator=(const Term& other);

    // Comparison operators
    bool operator<(const Term& other) const;
    bool operator<=(const Term& other) const;
    bool operator>(const Term& other) const;
    bool operator>=(const Term& other) const;

    // Operations required by the library.
    // friend bool operator= (Term &cT);
    friend bool operator== (const Term &cT1, const Term &cT2);
    friend bool operator!= (const Term &cT1, const Term &cT2);

    // Evaluation
    /** Evaluate this term on given data.
     * @param input     input data to process with this Term
     * @param inputLen  length of the inpu data (number of unsigned chars)
     * @return          1/0 if term is a subset of input
     *                  i.e. if every input bit corresponding to 1s in term is also 1
     */
    bool evaluate(const unsigned char * input, term_size_t inputLen) const;

    /**
     * Returns position of a particular bit w.r.t. term elements (POLY_GENOME_ITEM_TYPE).
     * Returns index of POLY_GENOME_ITEM_TYPE
     * @param bitIndex      bit index in term
     * @param termIndex     which term is the desired bit in?
     * @param termSize      what is the size of terms?
     * @return              term element position
     */
    static inline unsigned int elementIndexWithinVector(int bitIndex, int termIndex, unsigned int termSize) {
        return 1 + termIndex*termSize + (bitIndex/(8*sizeof(POLY_GENOME_ITEM_TYPE)));
    }

    /**
     * Returns position of a particular bit inside term element.
     * Returns bit position inside POLY_GENOME_ITEM_TYPE.
     * @param bitIndex      bit index in term
     * @return              its position inside particular memory unit (POLY_GENOME_ITEM_TYPE)
     */
    static inline unsigned int bitIndexWithinElement(int bitIndex) {
        return bitIndex % (8*sizeof(POLY_GENOME_ITEM_TYPE));
    }
};

// Pointer to the term.
typedef Term* PTerm;

inline bool operator== (const Term &cT1, const Term &cT2);
inline bool operator!= (const Term &cT1, const Term &cT2);

// Term comparator
struct TermComparator {
  bool operator() (const Term& lhs, const Term& rhs) const { return lhs.compareTo(rhs) == -1; }
};

// PTerm comparator
struct PTermComparator {
  bool operator() (const PTerm& lhs, const PTerm& rhs) const { return lhs->compareTo(rhs) == -1; }
};

#endif // _TERM_H
