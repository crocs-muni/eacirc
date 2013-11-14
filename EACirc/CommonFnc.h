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

/** function converting Chi^2 value to corresponding p-value
 * taken from http://www.codeproject.com/Articles/432194/How-to-Calculate-the-Chi-Squared-P-Value
 * note: do not use full implementation from above website, it is not precise enough for small inputs
 * @param Dof   degrees of freedom
 * @param Cv    Chi^2 value
 * @return      p-value
 */
double chisqr(int Dof, double Cv);

/** incomplete gamma function
 * taken from http://www.crbond.com/math.htm (Incomplete Gamma function, ported from Zhang and Jin)
 * @param a
 * @param x
 * @param gin
 * @param gim
 * @param gip
 * @return
 */
int incog(double a,double x,double &gin,double &gim,double &gip);

/** gamma function
 * taken from http://www.crbond.com/math.htm (Gamma function in C/C++ for real arguments, ported from Zhang and Jin)
 * returns 1e308 if argument is a negative integer or 0 or if argument exceeds 171.
 * @param x     argument
 * @return      \Gamma(argument)
 */
double gamma0(double x);

#endif
