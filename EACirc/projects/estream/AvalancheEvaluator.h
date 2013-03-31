#ifndef AVALANCHE_EVALUATOR_H
#define AVALANCHE_EVALUATOR_H

#include "evaluators/IEvaluator.h"
#include "EncryptorDecryptor.h"

class AvalancheEvaluator: public IEvaluator {
    //! pointer to project's encryptorDecryptor
    EncryptorDecryptor* m_encryptorDecryptor;
public:
    AvalancheEvaluator(EncryptorDecryptor* encryptorDecryptor);
    void evaluateCircuit(unsigned char*, unsigned char*, unsigned char*, int*, int*, int*);
    string shortDescription();
};

#endif
