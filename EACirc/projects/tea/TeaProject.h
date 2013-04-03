#ifndef TEAPROJECT_H
#define TEAPROJECT_H

#include "projects/IProject.h"

class TeaProject : public IProject {
public:
    TeaProject();
    ~TeaProject();
    string shortDescription() const;
    int generateTestVectors();
};

#endif // TEAPROJECT_H
