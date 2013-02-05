#ifndef ESTREAMPROJECT_H
#define ESTREAMPROJECT_H

#include "projects/IProject.h"
#include "EncryptorDecryptor.h"
#include "EstreamInterface.h"
#include "EstreamConstants.h"

class EstreamProject : public IProject {
private:
    unsigned char outputs[MAX_OUTPUTS];
    unsigned char inputs[MAX_INPUTS];
    int *numstats;
    EncryptorDecryptor* encryptorDecryptor;
    ESTREAM_SETTINGS estreamSettings;
    int getTestVector();
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
