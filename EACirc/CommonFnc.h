#ifndef COMMONFNC_H
#define COMMONFNC_H

#include <string>
#include "SSGlobals.h"

using namespace std;

int BYTE_ConvertFromHexStringToArray(string hexaString, unsigned char* pArray, unsigned char* pbArrayLen);
int BYTE_ConvertFromHexStringToArray(string hexaString, unsigned char* pArray, unsigned long* pbArrayLen);
int BYTE_ConvertFromArrayToHexString(unsigned char* pArray, unsigned long pbArrayLen, string* pHexaString);
void TrimLeadingSpaces(string& str);
void TrimTrailingSpaces(string& str);
double StringToDouble(string &s, bool failIfLeftoverChars = true);

#endif