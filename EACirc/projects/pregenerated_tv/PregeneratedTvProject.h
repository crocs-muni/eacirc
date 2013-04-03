#ifndef PREGENERATEDTVPROJECT_H
#define PREGENERATEDTVPROJECT_H

#include "projects/IProject.h"

class PregeneratedTvProject : public IProject {
private:
    //! stream with test vector file
    ifstream m_tvFile;
public:
    /** constructor, opens the test vector file for reading
      */
    PregeneratedTvProject();

    /** destructor, closes opened files
      */
    ~PregeneratedTvProject();

    string shortDescription() const;

    /** project initialization
      * - settings are read from test vector file
      * - settings are validated to match current global settings
      * @return status
      */
    int initializeProject();

    /** new set is read from file
      * @return status
      */
    int generateTestVectors();
};

#endif // PREGENERATEDTVPROJECT_H
