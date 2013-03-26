#ifndef CHECKER_H
#define CHECKER_H

#include "EACglobals.h"
#include "evaluators/IEvaluator.h"

class Checker {
    //! name of currently open file with test vectors
    string m_tvFilename;
    //! test vectors file structure
    ifstream m_tvFile;
    //! internat object status (error code)
    int m_status;
    //! allocator to use
    IEvaluator* m_evaluator;
    //! structure of main settings
    SETTINGS m_settings;
public:
    /** constructor
      * allocate GLOBALS and settings structure
      */
    Checker();

    /** destructor
      * delete evaluator, release GLOBALS
      */
    ~Checker();

    /** open file with test vectors
      * @param filename     name of file with test vectors
      */
    void setTestVectorFile(string filename);

    /** parse and check test vector file header
      */
    void loadTestVectorParameters();

    /** main method
      * - evaluate circuit on all vectors from file (sets of vectors)
      * - flush results to FILE_FITNESS_PROGRESS
      */
    void check();

    /** returns current error status
      * @return status
      */
    int getStatus() const;
};

#endif // CHECKER_H
