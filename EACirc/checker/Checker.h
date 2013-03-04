#ifndef CHECKER_H
#define CHECKER_H

#include "EACglobals.h"

class Checker {
    string m_tvFilename;
    istream m_tvFile;
    int m_status;
    int m_max_inputs;
    int m_max_outputs;
    //! structure of main settings
    SETTINGS m_settings;
public:
    Checker();
    ~Checker();
    int setTestVectorFile(string filename);
    int loadTestVectorParameters();
    int check();

    /** returns current error status
      * @return status
      */
    int getStatus() const;
};

int main();

#endif // CHECKER_H
