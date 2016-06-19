#include "CommonFnc.h"
//#include "EACglobals.h"
//#include "generators/IRndGen.h"
#include <bitset>
#include "logger.h"
#include <limits>
#include <set>
#include <cmath>

#ifndef M_PI // to resolve cmath constants problems
#define M_PI 3.141592653589793238462
#endif

using namespace std;

double CommonFnc::chisqr(int Dof, double Cv) {
    if (Cv < 0 || Dof < 1) {
        return 1;
    }
    double K = ((double)Dof) * 0.5;
    double X = Cv * 0.5;
    if (Dof == 2) {
        return exp(-1.0 * X);
    }
    double gin, gim, gip;
    incog(K, X, gin, gim, gip); // compute incomplete gamma function
    double PValue = gim;
    PValue /= gamma0(K); // divide by gamma function value
    return PValue;
}

int CommonFnc::incog(double a, double x, double& gin, double& gim, double& gip) {
    double xam, r, s, ga, t0;
    int k;

    if ((a < 0.0) || (x < 0))
        return 1;
    xam = -x + a * log(x);
    if ((xam > 700) || (a > 170.0))
        return 1;
    if (x == 0.0) {
        gin = 0.0;
        gim = gamma0(a);
        gip = 0.0;
        return 0;
    }
    if (x <= 1.0 + a) {
        s = 1.0 / a;
        r = s;
        for (k = 1; k <= 60; k++) {
            r *= x / (a + k);
            s += r;
            if (fabs(r / s) < 1e-15)
                break;
        }
        gin = exp(xam) * s;
        ga = gamma0(a);
        gip = gin / ga;
        gim = ga - gin;
    } else {
        t0 = 0.0;
        for (k = 60; k >= 1; k--) {
            t0 = (k - a) / (1.0 + k / (x + t0));
        }
        gim = exp(xam) / (x + t0);
        ga = gamma0(a);
        gin = ga - gim;
        gip = 1.0 - gim / ga;
    }
    return 0;
}

/* gamma function.
Algorithms and coefficient values from "Computation of Special
Functions", Zhang and Jin, John Wiley and Sons, 1996.
(C) 2003, C. Bond. All rights reserved.
taken from http://www.crbond.com/math.htm */
double CommonFnc::gamma0(double x) {
    int i, k, m;
    double ga, gr, r = 1.0, z;

    static double g[] = {1.0,
                         0.5772156649015329,
                         -0.6558780715202538,
                         -0.420026350340952e-1,
                         0.1665386113822915,
                         -0.421977345555443e-1,
                         -0.9621971527877e-2,
                         0.7218943246663e-2,
                         -0.11651675918591e-2,
                         -0.2152416741149e-3,
                         0.1280502823882e-3,
                         -0.201348547807e-4,
                         -0.12504934821e-5,
                         0.1133027232e-5,
                         -0.2056338417e-6,
                         0.6116095e-8,
                         0.50020075e-8,
                         -0.11812746e-8,
                         0.1043427e-9,
                         0.77823e-11,
                         -0.36968e-11,
                         0.51e-12,
                         -0.206e-13,
                         -0.54e-14,
                         0.14e-14};

    if (x > 171.0)
        return 1e308; // This value is an overflow flag.
    if (x == (int)x) {
        if (x > 0.0) {
            ga = 1.0; // use factorial
            for (i = 2; i < x; i++) {
                ga *= i;
            }
        } else
            ga = 1e308;
    } else {
        if (fabs(x) > 1.0) {
            z = fabs(x);
            m = (int)z;
            r = 1.0;
            for (k = 1; k <= m; k++) {
                r *= (z - k);
            }
            z -= m;
        } else
            z = x;
        gr = g[24];
        for (k = 23; k >= 0; k--) {
            gr = gr * z + g[k];
        }
        ga = 1.0 / (gr * z);
        if (fabs(x) > 1.0) {
            ga *= r;
            if (x < 0.0) {
                ga = -M_PI / (x * ga * sin(M_PI * x));
            }
        }
    }
    return ga;
}

double CommonFnc::KSGetCriticalValue(unsigned long sampleSize, unsigned significanceLevel) {
    if (sampleSize <= 35) {
        Logger::error() << "Too few samples for KS critical value (<=35)." << endl;
        return -1;
    }
    switch (significanceLevel) {
    case 10:
        return 1.224 / sqrt((double)sampleSize);
    case 5:
        return 1.358 / sqrt((double)sampleSize);
    case 1:
        return 1.628 / sqrt((double)sampleSize);
    default:
        Logger::error() << "Unusual significance level (not 1, 5, 10)." << endl;
        return -1;
    }
}

double CommonFnc::KSUniformityTest(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    // sanity check
    if (samples[samples.size() - 1] > 1 || samples[0] < 0) {
        Logger::error() << "Cannot run K-S test, data out of range." << endl;
    }

    double test_statistic = 0;
    double temp = 0;
    double N = samples.size();

    for (int i = 0; i < N; i++) {
        temp = max(samples[i] - i / N, (i + 1) / N - samples[i]);
        test_statistic = max(test_statistic, temp);
    }

    return test_statistic;
}

double CommonFnc::zscore(double observed, double expected, double samples){
    return (observed-expected) / sqrt((expected*(1-expected))/samples);
}

double CommonFnc::ucrit(double alpha){
    if (alpha == 0.1) {
        return 1.281522;

    } else if (alpha == 0.05){
        return 1.644854;

    } else if (alpha == 0.025){
        return 1.959964;

    } else if (alpha == 0.01){
        return 2.326348;

    } else if (alpha == 0.005){
        return 2.575829;

    } else if (alpha == 0.001){
        return 3.090232;

    } else if (alpha == 0.0001){
        return 3.719016;

    } else {
        throw std::out_of_range("u(alpha) not implemented for this value");
    }
}

double CommonFnc::binomialU(double n, double successCtr, double p0){
    return (successCtr - n*p0) / sqrt((n*p0)*(1-p0));
}

long long CommonFnc::getFileSize(std::string filename) {
    ifstream myFile(filename, ios::binary); // get filesize V2
    myFile.seekg(0, ios::end);
    auto len = myFile.tellg();
    myFile.close();
    return (unsigned long long) len;
}
