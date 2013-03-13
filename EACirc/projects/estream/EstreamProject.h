#ifndef ESTREAMPROJECT_H
#define ESTREAMPROJECT_H

#include "projects/IProject.h"
#include "EncryptorDecryptor.h"
#include "EstreamInterface.h"
#include "EstreamConstants.h"

class EstreamProject : public IProject {
private:
    //! arrays for single final test vector
    unsigned char* m_tvOutputs;
    unsigned char* m_tvInputs;
    //! temporary arrays for initial and encrypted-decrypted plaintext (should be the same)
    u8* m_plaintextIn;
    u8* m_plaintextOut;
    //! array used for ballacing test vectors
    int* m_numVectors;
    //! object providing eStream cipher facilities
    EncryptorDecryptor* m_encryptorDecryptor;
    //! settings for eStream project
    ESTREAM_SETTINGS m_estreamSettings;

    /** prepare single test vector
      * @return status
      */
    int getTestVector();

    /** generate binary output of ciphertext to cout
      * - used to generate data stream for statistical testing (dieharder, etc.)
      * - at least the number of bytes specified in config file is generated (rounded to nearest set)
      * - generation is infinite if streamSize is set to 0
      * - after generation, whole program ends with status INTENTIONAL_EXIT
      * @return status
      */
    int generateCipherDataStream();

    /** set plaintext according to loaded settings (key and IV are set via EncryptorDecryptor)
      * @return status
      */
    int setupPlaintext();
public:
    /** constructor, memory allocation
      */
    EstreamProject();

    /** destructor, memory deallocation
      */
    ~EstreamProject();

    string shortDescription() const;

    /** load project-specific configuration from config file
      * @param pRoot    allocated project config file XML root (corresponding to EACIRC element)
      * @return status
      */
    int loadProjectConfiguration(TiXmlNode* pRoot);

    /** project initialization
      * - create EncryptorDecryptor
      * - create project-specifiec header in test vector file (if needed)
      * @return status
      */
    int initializeProject();

    /** init IV and key, if set to be initialized only once
      * @return status
      */
    int initializeProjectState();

    /** save state - key, IV (if loadable)
      * @param pRoot    allocated project state XML root
      * @return status
      */
    int saveProjectState(TiXmlNode* pRoot) const;

    /** load state - key, IV (if applicable)
      * @param pRoot    allocated project state XML root
      * @return status
      */
    int loadProjectState(TiXmlNode* pRoot);

    /** prepares complete test vector set set
      * @return status
      */
    int generateTestVectors();
};

#endif // ESTREAMPROJECT_H
