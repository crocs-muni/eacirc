#ifndef IPROJECT_H
#define IPROJECT_H

#include "EACglobals.h"
#include "XMLProcessor.h"
#include "generators/IRndGen.h"

class IProject {
protected:
    //! project type, see EACirc constants
    int m_type;

private:
    /** generate new set of test vectors
      * -> generated vectors to be saved in globally reachable memory (pGlobals->testVectors)
      * implementation in project required!
      * @return status
      */
    virtual int generateTestVectors() = 0;

    /** save current test vectors to file
      * @return status
      */
    int saveTestVectors() const;

    /** initialize project itself
      * called once, after configuration loading
      * called REGARDLESS of recommencing/not recommencong computation
      * project can here write own configuration notes to test vectors file
      * called by initializeProjectMain();
      * default implemetation: do nothing
      * @return status;
      */
    virtual int initializeProject();

    /** save project state
      * - root node with project type and description is prepared by framework
      * - add project details into the tree
      * - project is set as 'loadable' by default, you can change this
      * - called by saveProjectStateMain()
      * note: if project state is loadable, reset the appropriate attribute!
      *       (not loadable is default)
      * default implementation: do nothing
      * @param pRoot    allocated root node for project
      * @return status
      */
    virtual int saveProjectState(TiXmlNode* pRoot) const;

    /** load project state
      * - project constant and loadability checks are performed by framework
      * - load your project-specific settings
      * - called by loadProjectStateMain()
      * default implementation: do nothing
      * @param pRoot    allocated XML root of project state
      * @return status
      */
    virtual int loadProjectState(TiXmlNode* pRoot);

public:
    /** general project constructor, sets project type
      * - if enabled, creates header for test vectors file
      * @param type     project constant
      */
    IProject(int type);

    /** general project destructor
      */
    virtual ~IProject();

    /** short textual description of the project
      * implementation in project required!
      * @return description
      */
    virtual string shortDescription() const = 0;

    /** load project-specific configuration
      * default implementation: load no configuration
      * @param pRoot    parsed XML tree with configuration (root=EACIRC)
      * @return status
      */
    virtual int loadProjectConfiguration(TiXmlNode* pRoot);

    /** initialize project
      * - called once, after configuration loading
      * - calls project initialization (virtual function initializeProject();)
      * - make header structure in test vector file
      * - called REGARDLESS of recommencing/not recommencong computation
      * @return status;
      */
    int initializeProjectMain();

    /** initialize project state (called just once, before evolution)
      * NOT called at all, when computation is recommenced
      * default implementation: do nothing
      * @return status
      */
    virtual int initializeProjectState();

    /** load project state (previously saved by this project)
      * - checks project constant
      * - checks loadability
      * - calles virtual project state loading method
      * @param pRoot    parsed XML sructure with project state
      * @return status
      */
    int loadProjectStateMain(TiXmlNode* pRoot);

    /** save current project state
      * @return allocated XML tree with project state
      *         CALLER responsible for freeing!
      * default implementation: save project constant and description, make it non-loadable
      */
    TiXmlNode* saveProjectStateMain() const;

    /** generate new test vectors and save them if required
      * @return status
      */
    int generateAndSaveTestVectors();

    /** constatnt of active project
      * @return project constant
      */
    int getProjectType() const;

    /** static function to get project instance
      */
    static IProject* getProject(int projectType);
};

#endif // IPROJECT_H
