#ifndef FILEDISTINGUISHERPROJECT_H
#define FILEDISTINGUISHERPROJECT_H

#include "projects/IProject.h"
#include "fileDistinguisherConstants.h"

class FileDistinguisherProject : public IProject {
    //! streams with open files
    ifstream m_files[FILEDIST_NUMBER_OF_FILES];
    //! offset for reading from files (max 4GB)
    unsigned long m_readOffsets[FILEDIST_NUMBER_OF_FILES];
    //! settings for file distinguisher project
    FILE_DISTINGUISHER_SETTINGS m_fileDistSettings;
public:
    /** constructor, clean attributes
      */
    FileDistinguisherProject();

    /** destructor, closing files
      */
    ~FileDistinguisherProject();

    string shortDescription() const;

    /** load project-specific configuration from config file
      * @param pRoot    allocated project config file XML root (corresponding to EACIRC element)
      * @return status
      */
    int loadProjectConfiguration(TiXmlNode* pRoot);

    /** project initialization
      * - open files
      * - get file size
      * @return status
      */
    int initializeProject();

    /** get random read offset for files, if needed
      * @return status
      */
    int initializeProjectState();

    /** save state - filenames, file sizes, read offset
      * @param pRoot    allocated project state XML root
      * @return status
      */
    int saveProjectState(TiXmlNode* pRoot) const;

    /** load state - file sizes, read offset
      * @param pRoot    allocated project state XML root
      * @return status
      */
    int loadProjectState(TiXmlNode* pRoot);

    /** create headers in testVectorFile and humanReadableTestVectorFile
      * - called only if saving test vectors, no need to doublecheck
      * @return status
      */
    int createTestVectorFilesHeaders() const;

    /** prepares complete test vector set set
      * @return status
      */
    int generateTestVectors();
};

#endif // FILEDISTINGUISHERPROJECT_H
