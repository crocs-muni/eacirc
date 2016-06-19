#ifndef COMMONFNC_H
#define COMMONFNC_H

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>


namespace CommonFnc {

/** convert given binary array to hexa
 * @param data          array to convert
 * @param dataLength    number of uchars in data
 */
std::string arrayToHexa(unsigned char* data, unsigned int dataLength);

/** convert given binary array to hexa
* @param hexa          string with hexadecimal aray value
* @param dataLength    number of uchars in data
* @param data          array to return (must be pre-allocated)
*/
int hexaToArray(std::string hexa, unsigned int dataLength, unsigned char* data);

/** remove file
  * - remove file from system according to parameter
  * - errors are output to logger as warnings
  * @param filename     file to delete
  */
void removeFile(std::string filename);

long long getFileSize(std::string filename);

template <typename T> std::string toString(T value) {
    std::ostringstream ss;
    ss << std::left << std::dec;
    ss << value;
    return ss.str();
}

/** function converting Chi^2 value to corresponding p-value
 * taken from
 * http://www.codeproject.com/Articles/432194/How-to-Calculate-the-Chi-Squared-P-Value
 * note: do not use full implementation from above website, it is not precise
 * enough for small inputs
 * @param Dof   degrees of freedom
 * @param Cv    Chi^2 value
 * @return      p-value
 */
double chisqr(int Dof, double Cv);

/** incomplete gamma function
 * taken from http://www.crbond.com/math.htm (Incomplete Gamma function, ported
 * from Zhang and Jin)
 * @param a
 * @param x
 * @param gin
 * @param gim
 * @param gip
 * @return
 */
int incog(double a, double x, double& gin, double& gim, double& gip);

/** gamma function
 * taken from http://www.crbond.com/math.htm (Gamma function in C/C++ for real
 * arguments, ported from Zhang and Jin)
 * returns 1e308 if argument is a negative integer or 0 or if argument exceeds
 * 171.
 * @param x     argument
 * @return      \Gamma(argument)
 */
double gamma0(double x);

/** returns critical value for KS test
 * @param sampleSize            number of samples
 * @param significanceLevel     alpha in percents (only 1, 5, 10 supported)
 * @return critical value
 */
double KSGetCriticalValue(unsigned long sampleSize, unsigned significanceLevel);

/** Kolmogorov-Smirnov uniformity test
 * - tests uniformity distribution on [0,1]
 * - works only for number from range [0,1]
 * - idea taken from http://www.jstatsoft.org/v08/i18/paper
 * (Evaluating Kolmogorov’s Distribution by George Masaglia et alii)
 * @param samples       vector of observer values -- will be sorted!
 * @return KS test statistic value
 */
double KSUniformityTest(std::vector<double>& samples);

/**
 * Distance of a observed from expected in standard deviations for Normal distribution
 */
double zscore(double observed, double expected, double samples);

/**
 * Binomial distribution testing statistics.
 * Returns U. Reject if |U| >= u(alpha/2)
 *
 * @param n number of samples
 * @param sucessCtr number of success events
 * @param p0 testing probability of success event.
 */
double binomialU(double n, double successCtr, double p0);

/**
 * Compute mean of vector values
 */
template<typename T>
double computeMean(std::vector<T> data){
    return std::accumulate(data.begin(), data.end(), 0) / (double)data.size();
}

/**
 * Compute standard deviation on input vector.
 */
template<typename T>
double computeStddev(std::vector<T> data, double mean){
    auto size = data.size();
    double var = 0;
    for(auto i = 0; i < size; i++ ) {
        var += (data[i] - mean) * (data[i] - mean);
    }
    var /= size;
    return sqrt(var);
}

} // namespace CommonFnc

#endif
