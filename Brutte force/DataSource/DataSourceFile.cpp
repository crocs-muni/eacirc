//
// Created by Dusan Klinec on 19.06.16.
//

#include "DataSourceFile.h"
#include "../CommonFnc.h"

using namespace std;
DataSourceFile::DataSourceFile(std::string fileName) {
    this->fileName = fileName;
    this->fileSize = CommonFnc::getFileSize(fileName);
    this->in.open(fileName, ios::binary);
}

long long DataSourceFile::getAvailableData() {
    return this->fileSize;
}

void DataSourceFile::read(char *buffer, size_t size) {
    this->in.read(buffer, (streamsize) size);
}

std::string DataSourceFile::desc() {
    return this->fileName;
}


