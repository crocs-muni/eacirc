#include "statistics.h"
#include <cmath>

/** gamma function
 * taken from http://www.crbond.com/math.htm (Gamma function in C/C++ for real
 * arguments, ported from Zhang and Jin)
 * returns 1e308 if argument is a negative integer or 0 or if argument exceeds
 * 171.
 * @param x     argument
 * @return      Gamma(argument)
 */
double gamma0(double x) {
    /* gamma function.
    Algorithms and coefficient values from "Computation of Special
    Functions", Zhang and Jin, John Wiley and Sons, 1996.
    (C) 2003, C. Bond. All rights reserved.
    taken from http://www.crbond.com/math.htm */
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
    if (x == int(x)) {
        if (x > 0.0) {
            ga = 1.0; // use factorial
            for (i = 2; i < x; i++) {
                ga *= i;
            }
        } else
            ga = 1e308;
    } else {
        if (std::fabs(x) > 1.0) {
            z = std::fabs(x);
            m = int(z);
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
        if (std::fabs(x) > 1.0) {
            ga *= r;
            if (x < 0.0) {
                ga = -M_PI / (x * ga * std::sin(M_PI * x));
            }
        }
    }
    return ga;
}

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
static int incog(double a, double x, double& gin, double& gim, double& gip) {
    double xam, r, s, ga, t0;
    int k;

    if ((a < 0.0) || (x < 0))
        return 1;
    xam = -x + a * std::log(x);
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
            if (std::fabs(r / s) < 1e-15)
                break;
        }
        gin = std::exp(xam) * s;
        ga = gamma0(a);
        gip = gin / ga;
        gim = ga - gin;
    } else {
        t0 = 0.0;
        for (k = 60; k >= 1; k--) {
            t0 = (k - a) / (1.0 + k / (x + t0));
        }
        gim = std::exp(xam) / (x + t0);
        ga = gamma0(a);
        gin = ga - gim;
        gip = 1.0 - gim / ga;
    }
    return 0;
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
static double chisqr(int Dof, double Cv) {
    if (Cv < 0 || Dof < 1) {
        return 1;
    }
    double K = (double(Dof)) * 0.5;
    double X = Cv * 0.5;
    if (Dof == 2) {
        return std::exp(-1.0 * X);
    }
    double gin, gim, gip;
    incog(K, X, gin, gim, gip); // compute incomplete gamma function
    double PValue = gim;
    PValue /= gamma0(K); // divide by gamma function value
    return PValue;
}

double two_sample_chisqr::_compute() const {
    // using two-smaple Chi^2 test
    // (http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm)

    double k1 = 1;
    double k2 = 1;
    double chisqr_value = 0;
    int dof = 0;

    for (unsigned i = 0; i != _histogram_a.size(); ++i) {
        auto sum = _histogram_a[i] + _histogram_b[i];
        if (sum > 5) {
            dof++;
            chisqr_value += std::pow(k1 * _histogram_a[i] - k2 * _histogram_b[i], 2) / sum;
        }
    }
    dof--; // last category is fully determined by others

    return chisqr(dof, chisqr_value);
}

#include <algorithm>
#include <stdexcept>

double ks_uniformity_test::_compute_critical_value(std::size_t size, unsigned significance_level) {
    if (size <= 35)
        throw std::runtime_error("Too few samples for KS critical value (<=35).");

    switch (significance_level) {
    case 10:
        return 1.224 / std::sqrt(double(size));
    case 5:
        return 1.358 / std::sqrt(double(size));
    case 1:
        return 1.628 / sqrt(double(size));
    default:
        throw std::runtime_error("Significance level must be 1, 5, or 10.");
    }
}

double ks_uniformity_test::_compute_uniformity_test(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());

    if (samples.front() < 0 || samples.back() > 1)
        throw std::out_of_range("Cannot run K-S test, data out of range.");

    double test_statistic = 0;
    double n = samples.size();

    for (std::size_t i = 0; i != samples.size(); ++i) {
        double temp = std::max(samples[i] - i / n, (i + 1) / n - samples[i]);
        test_statistic = std::max(test_statistic, temp);
    }
    return test_statistic;
}
