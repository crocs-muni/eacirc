#ifndef HASHER_H
#define HASHER_H

#include "EACglobals.h"
#include "Sha3Constants.h"
#include "Sha3Interface.h"
#include "XMLProcessor.h"

class Hasher {
    /** array of used hash functions
      *   0: use algorithm1 with its settings
      *   1: use algorithm2 with its settings
      */
    Sha3Interface* m_hashFunctions[2];

    //! array with computed hashes
    unsigned char* m_hashOutputs[2];
    //! array with hash output lengths
    int m_hashOutputLengths[2];
    //! counters for hashed data
    unsigned long m_counters[2];
    //! number of bits of computed hashes already used in test vectors
    int m_usedBytes[2];
    //! initialization successful?
    bool m_initSuccess;

    /** if number of rounds > -1, log info about number of rounds not supported
      */
    void checkNumRounds(int numRounds, int algorithmConstant);

public:
    /** constructor
      * - allocates hash functions according to loaded settings
      * - create header for human readable test vector file
      */
    Hasher();

    /** Was initialization successful?
     * @return success
     */
    bool initSuccess() const;

    /** destructor, release memory
      */
    ~Hasher();

    /** set counters to initial value
      */
    int initializeState();

    /** save state of hasher
      * - current hash outputs, lengths, counters, usedBytes
      * @param pRoot    allocated XML element 'hasher' to be filled with state
      * @return status
      */
    int saveHasherState(TiXmlNode* pRoot) const;

    /** load state of hasher
      * - current hash outputs, lengths, counters, usedBytes
      * @param pRoot    XML element 'hasher'
      * @return status
      */
    int loadHasherState(TiXmlNode* pRoot);

    /** prepare single test vector for given algorithm (according to plaintext type)
      * @param algorithmNumber      1 for algorithm_1, 2 for algorithm_2
      * @param tvInputs             array to store test vector inputs
      * @param tvOutputs            array to store test vector outputs
      * @return status
      */
    int getTestVector(int algorithm, unsigned char* tvInputs, unsigned char *tvOutputs);
};

#endif // HASHER_H
