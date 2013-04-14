#ifndef FILESPROJECT_H
#define FILESPROJECT_H

#include "projects/IProject.h"
#include "filesConstants.h"

class FilesProject : public IProject {
    //! arrays for single final test vector
    unsigned char* m_tvOutputs;
    unsigned char* m_tvInputs;
    //! array used for ballacing test vectors
    int* m_numVectors;
    //! streams with open files
    ifstream m_files[FILES_NUMBER_OF_FILES];
    //! offset for reading from files (max 4GB)
    unsigned long m_readOffsets[FILES_NUMBER_OF_FILES];
    //! settings for file distinguisher project
    FILES_SETTINGS m_fileDistSettings;

    /** prepare single test vector
      * test vector is saved to m_tvOutputs/m_tvInputs
      * @return status
      */
    int prepareSingleTestVector();

    /** read stream from file
      * - revind file, if necessary
      * @param fileNumber       from which file to read?
      * @param length           how many bytes?
      * @param data             where to store data (should already be allocated)
      * @return status
      */
    int getStreamFromFile(int fileNumber, unsigned long length, unsigned char* data);

public:
    /** constructor, clean attributes, allocate memory
      */
    FilesProject();

    /** destructor, closing files, deallocate memory
      */
    ~FilesProject();

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

#endif // FILESPROJECT_H
