#ifndef ESTREAMPROJECT_H
#define ESTREAMPROJECT_H

#include "projects/IProject.h"

class EstreamProject : public IProject {
public:
    EstreamProject();
    ~EstreamProject();
    string shortDescription() const;
    int generateTestVectors();
};

#endif // ESTREAMPROJECT_H
