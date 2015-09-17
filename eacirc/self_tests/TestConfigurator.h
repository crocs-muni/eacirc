#ifndef TESTCONFIGURATOR_H
#define TESTCONFIGURATOR_H

#include "EACglobals.h"
#include <queue>

#define BACKUP_SUFFIX ".2"

class TestConfigurator {
    //! project currently tested
    int m_currentProject;
    //! array of projects yet to test
    queue<int> m_projects;

public:
    /** constructor, fill project constants array with all projects
     */
    TestConfigurator();

    ~TestConfigurator();

    /** set new project as current
     * @return is project set?
     */
    bool nextProject();

    /** add all projects for testing
     */
    void addAllProjects();

    /** add additional project to testing
     * @param projectType project to add 
     */
    void addProject(int projectType);

    /**
     * @return currently tested project
     */
    int getCurrentProject();

    /** save default testing configuration for given project in config file
     * argumentless version prepares m_currentProject
     * @param projectType constant (current project by default);
     * @return status
     */
    void prepareConfiguration(int projectType) const;
    void prepareConfiguration() const;

    /** backup given file (rename with BACKUP-SUFFIX)
      * any previous version of backup-file is overwritten
      * @param filename
      */
    void backupFile(string filename);

    /** backup common result files
      * FILE_GALIB_SCORES, FILE_FITNESS_PROGRESS, FILE_STATE, FILE_POPULATION
      */
    void backupResults();

    /** compare common result files (commom filename and backup version ending with BACKUP_SUFFIX)
      * FILE_GALIB_SCORES, FILE_FITNESS_PROGRESS, FILE_STATE, FILE_POPULATION
      */
    void compareResults() const;

    /** compare contents of given files line by line
      * if files differ in more than 5 lines, comparation is terminated
      *
      * @param filename1
      * @param filename2
      */
    void compareFilesByLine(string filename1, string filename2) const;

    /** run EACirc computation
      * perform all nedded steps (create, load, initialize, prepare, run)
      * check final status for STAT_OK
      */
    void runEACirc() const;

    /** constant with main testing configuration
     * without xml header, without root EACIRC element
     */
    static string mainConfiguration;

    /** Tests the equality of double precision floating point numbers.
    * Comparing floating point numbers is rather involved.
    * See for example https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    * For simplicity, we use relative epsilon.
    * @param x
    * @param y
    * @return true for x ~= y
    */
    static bool floatingPointEqual(double x, double y);
};

#endif // TESTCONFIGURATOR_H
