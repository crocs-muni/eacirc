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

    /** start infinite loop and generate binary output of ciphertext to cout
      * - used to generate data stream for statistical testing (dieharder, etc.)
      * - prevents evolution from starting
      * - get program to state, where it must be killed externally!
      * @return status
      */
    int generateCipherDataStream();

    /** set plaintext according to loaded settings (key and IV are set via EncryptorDecryptor)
      * @return status
      */
    int setupPlaintext();
public:
    EstreamProject();
    ~EstreamProject();
    string shortDescription() const;
    int loadProjectConfiguration(TiXmlNode* pRoot);
    int initializeProject();
    int initializeProjectState();
    int generateTestVectors();
};

#endif // ESTREAMPROJECT_H
