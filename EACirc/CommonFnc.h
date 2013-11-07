#ifndef COMMONFNC_H
#define COMMONFNC_H

#include <string>
#include <iostream>
#include <sstream>

using namespace std;

int BYTE_ConvertFromHexStringToArray(string hexaString, unsigned char* pArray, unsigned char* pbArrayLen);
int BYTE_ConvertFromHexStringToArray(string hexaString, unsigned char* pArray, unsigned long* pbArrayLen);
int BYTE_ConvertFromArrayToHexString(unsigned char* pArray, unsigned long pbArrayLen, string* pHexaString);
void TrimLeadingSpaces(string& str);
void TrimTrailingSpaces(string& str);
double StringToDouble(string &s, bool failIfLeftoverChars = true);

/** copy file
  * - binary mode
  * - if destination file exists, it will be overwritten
  * @param source           what file to copy
  * @param destination      where to copy file
  * @return status
  */
int copyFile(string source, string destination);

template < typename T >
string toString(T value) {
    stringstream ss;
    ss << left << dec;
    ss << value;
    return ss.str();
}

// from http://www.codeproject.com/Articles/432194/How-to-Calculate-the-Chi-Squared-P-Value
double chisqr(int Dof, double Cv);

// from http://www.codeproject.com/Articles/432194/How-to-Calculate-the-Chi-Squared-P-Value
double igf(double S, double Z);

#endif
