#ifndef ESTREAMPROJECT_H
#define ESTREAMPROJECT_H

#include "projects/IProject.h"
#include "EncryptorDecryptor.h"
#include "estreamInterface.h"
#include "EstreamConstants.h"

extern ESTREAM_SETTINGS* pEstreamSettings;

class EstreamProject : public IProject {
private:
    unsigned char outputs[MAX_OUTPUTS];
    unsigned char inputs[MAX_INPUTS];
    int *numstats;
    EncryptorDecryptor* encryptorDecryptor;
    ESTREAM_SETTINGS estreamSettings;
    void getTestVector();
public:
    EstreamProject();
    ~EstreamProject();
    string shortDescription() const;
    int loadProjectConfiguration(TiXmlNode* pRoot);
    int generateTestVectors();
};

#endif // ESTREAMPROJECT_H
