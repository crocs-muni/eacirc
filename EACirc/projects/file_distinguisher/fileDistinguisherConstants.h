#ifndef FILEDISTINGUISHERCONSTANTS_H
#define FILEDISTINGUISHERCONSTANTS_H

//! total number of files in project
#define FILEDIST_NUMBER_OF_FILES       2

struct FILE_DISTINGUISHER_SETTINGS {
    string filenames[2];
    bool useFixedInitialOffset;
    unsigned long initialOffsets[2];
    unsigned long fileSizes[2];
    FILE_DISTINGUISHER_SETTINGS(void) {
        useFixedInitialOffset = false;
        for (int i = 0; i < 2; i++) {
            filenames[i] = "";
            initialOffsets[i] = 0;
            fileSizes[i] = 0;
        }
    }
};

extern FILE_DISTINGUISHER_SETTINGS* pFileDistSettings;

#endif // FILEDISTINGUISHERCONSTANTS_H
