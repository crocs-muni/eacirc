#ifndef FILESCONSTANTS_H
#define FILESCONSTANTS_H

//! total number of files in project
#define FILES_NUMBER_OF_FILES       2

// usage types
#define FILES_DISTINGUISHER         401

struct FILES_SETTINGS {
    int usageType;
    string filenames[2];
    bool ballancedTestVectors;
    bool useFixedInitialOffset;
    unsigned long initialOffsets[2];
    unsigned long fileSizes[2];
    FILES_SETTINGS(void) {
        usageType = -1;
        ballancedTestVectors = false;
        useFixedInitialOffset = false;
        for (int i = 0; i < 2; i++) {
            filenames[i] = "";
            initialOffsets[i] = 0;
            fileSizes[i] = 0;
        }
    }
};

extern FILES_SETTINGS* pFileDistSettings;

#endif // FILESCONSTANTS_H
