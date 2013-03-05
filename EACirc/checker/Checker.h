#ifndef CHECKER_H
#define CHECKER_H

#include "EACglobals.h"
#include "evaluators/IEvaluator.h"

class Checker {
    string m_tvFilename;
    ifstream m_tvFile;
    int m_status;
    int m_max_inputs;
    int m_max_outputs;
    IEvaluator* m_evaluator;
    //! structure of main settings
    SETTINGS m_settings;
public:
    Checker();
    ~Checker();
    void setTestVectorFile(string filename);
    void loadTestVectorParameters();
    void check();

    /** returns current error status
      * @return status
      */
    int getStatus() const;
};

#endif // CHECKER_H
