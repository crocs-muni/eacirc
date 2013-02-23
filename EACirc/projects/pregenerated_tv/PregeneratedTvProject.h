#ifndef PREGENERATEDTVPROJECT_H
#define PREGENERATEDTVPROJECT_H

#include "projects/IProject.h"

class PregeneratedTvProject : public IProject {
private:
    //! stream with test vector file
    ifstream m_tvFile;
public:
    PregeneratedTvProject();
    ~PregeneratedTvProject();
    string shortDescription() const;
    int initializeProject();
    int generateTestVectors();
};

#endif // PREGENERATEDTVPROJECT_H
