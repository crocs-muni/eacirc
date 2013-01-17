#ifndef SHA3PROJECT_H
#define SHA3PROJECT_H

#include "projects/IProject.h"

class Sha3Project : public IProject {
public:
    Sha3Project();
    ~Sha3Project();
    string shortDescription() const;
    int generateTestVectors();
};

#endif // SHA3PROJECT_H
