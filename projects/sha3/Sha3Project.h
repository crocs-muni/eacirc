#ifndef SHA3PROJECT_H
#define SHA3PROJECT_H

#include "projects/IProject.h"
#include "Hasher.h"

class Sha3Project : public IProject {
    //! arrays for single final test vector
    unsigned char* m_tvOutputs;
    unsigned char* m_tvInputs;
    //! array used for ballacing test vectors
    int* m_numVectors;
    //! object providing SHA-3 algorithm facilities
    Hasher* m_hasher;
    //! settings for SHA-3 project
    SHA3_SETTINGS m_sha3Settings;

    /** prepare single test vector
      * test vector is saved to m_tvOutputs/m_tvInputs
      * @return status
      */
    int prepareSingleTestVector();

    /** generate binary output of hashed data to file (for each non-random stream)
      * - used to generate data stream for statistical testing (dieharder, etc.)
      * - at least the number of bytes specified in config file is generated (rounded up to nearest set)
      * - if streamSize is set to 0, generation is inifinte and is sent to cout instead of file
      * - after generation, whole program ends with status INTENTIONAL_EXIT
      * @return status
      */
    int generateHashDataStream();

public:
    /** constructor, memory allocation
      */
    Sha3Project();

    /** destructor, memory deallocation
      */
    ~Sha3Project();

    string shortDescription() const;

    /** return configuration used for testing
     * @return testing configuration for project (project root)
     */
    static string testingConfiguration();

    /** load project-specific configuration from config file
      * @param pRoot    allocated project config file XML root (corresponding to EACIRC element)
      * @return status
      */
    int loadProjectConfiguration(TiXmlNode* pRoot);

    /** project initialization
      * - create Hasher
      * - create project-specifiec header in test vector files (if needed)
      * - allocated project specific evaluator (if used)
      * @return status
      */
    int initializeProject();

    /** set counters in hasher
      * @return status
      */
    int initializeProjectState();

    /** save state - hashes, counters, ...
      * @param pRoot    allocated project state XML root
      * @return status
      */
    int saveProjectState(TiXmlNode* pRoot) const;

    /** load state - hashes, counters, ...
      * @param pRoot    allocated project state XML root
      * @return status
      */
    int loadProjectState(TiXmlNode* pRoot);

    /** create headers in testVectorFile and humanReadableTestVectorFile
      * @return status
      */
    int createTestVectorFilesHeaders() const;

    /** prepares complete test vector set set
      * @return status
      */
    int generateTestVectors();
};

#endif // SHA3PROJECT_H
