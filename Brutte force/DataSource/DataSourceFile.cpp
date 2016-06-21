//
// Created by Dusan Klinec on 19.06.16.
//

#include "DataSourceFile.h"
#include "../CommonFnc.h"

using namespace std;
DataSourceFile::DataSourceFile(std::string fileName) {
    this->m_fileName = fileName;
    this->m_fileSize = CommonFnc::getFileSize(fileName);
    this->m_in.open(fileName, ios::binary);
}

long long DataSourceFile::getAvailableData() {
    return this->m_fileSize;
}

void DataSourceFile::read(char *buffer, size_t size) {
    this->m_in.read(buffer, (streamsize) size);
}

std::string DataSourceFile::desc() {
    return this->m_fileName;
}


