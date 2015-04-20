#ifndef FINISHERS_H
#define FINISHERS_H

#include "EACglobals.h"
#include "GAGenome.h"

namespace Finishers {

    /**
     * @brief Output average fitness values lo log.
     * AvgAvg = average over average values in selected generations
     * AvgMax = average over maximum values in selected generations
     * AvgMin = average over minimum values in selected generations
     */
    void avgFitnessFinisher();

    /**
     * @brief Test p-values uniformity using KS-test.
     * Kolmogorov-Smirnov test for the p-values, applicable only for categories evaluator.
     */
    void ksUniformityTestFinisher();

    /**
     * @brief Output selected genome to file.
     * - output in all available formats
     * - try to post-process and output again (if post-processing successful)
     * @param genome    genome to output
     */
    void outputCircuitFinisher(GAGenome& genome);
}

#endif // FINISHERS_H
