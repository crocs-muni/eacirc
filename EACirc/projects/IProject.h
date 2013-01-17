#ifndef IPROJECT_H
#define IPROJECT_H

#include "EACglobals.h"
#include "XMLProcessor.h"

class IProject {
protected:
    //! project type, see EACirc constants
    int m_type;
public:
    IProject(int type);
    virtual ~IProject();
    virtual string shortDescription() const = 0;
    virtual int loadProjectConfiguration(TiXmlNode* pRoot);
    virtual int generateTestVectors() = 0;
    virtual TiXmlNode* saveProjectState() const;
    virtual int loadProjectState(TiXmlNode* pRoot);
    static IProject* getProject(int projectType);
};

#endif // IPROJECT_H
