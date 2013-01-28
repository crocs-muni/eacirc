#ifndef IPROJECT_H
#define IPROJECT_H

#include "EACglobals.h"
#include "XMLProcessor.h"

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

public:
    /** general project constructor, sets project type
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

    /** initialize project (called just once, before evolution)
      * default implementation: do nothing
      * @return status
      */
    virtual int initialzeProjectState();

    /** load project state (previously saved by this project)
      * default implementation: check project constant, load nothing
      * @param pRoot    parsed XML sructure with project state
      * @return status
      */
    virtual int loadProjectState(TiXmlNode* pRoot);

    /** save current project state
      * @return allocated XML tree with project state
      *         CALLER responsible for freeing!
      * default implementation: save project constant
      */
    virtual TiXmlNode* saveProjectState() const;

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
