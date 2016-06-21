//
// Created by Dusan Klinec on 16.06.16.
//

#include "mainBi.h"

#include <iostream>
#include "dynamic_bitset.h"
#include "bit_array.h"
#include "Term.h"
#include "bithacks.h"
#include <ctime>
#include <random>
#include <fstream>
#include <unordered_map>
#include "CommonFnc.h"
#include "finisher.h"
#include "logger.h"
#include "TermGenerator.h"
#include "DataSource/DataSourceFile.h"
#include "DataSource/DataSourceAES.h"
#include "DataSource/DataSourceMD5.h"
#include "DataSource/DataSourceRC4.h"
#include "DataSource/DataSourceRC4Column.h"
#include "DataSource/DataSourceSystemRandom.h"
#include <algorithm>
#include <string>
#include <iomanip>
#include <random>
#include <memory>

#ifdef BOOST
#include <boost/math/distributions/students_t.hpp>
#endif

//
// Some definitions.
//
#define TERM_WIDTH_BYTES 16
#define TERM_DEG 3
//#define DUMP_FILES 1

// Do not edit.
#define TERM_WIDTH (TERM_WIDTH_BYTES*8)
// Combination(TERM_WIDTH, TERM_DEG) for C(128,3)
#define TERM_NUMBER 341376
typedef std::vector<int> termRep;


using namespace std;
#ifdef BOOST
using namespace boost::math;
#endif

int initState(){
    return -1;
}

template<typename T>
void histogram(vector<T> data, unsigned long bins, bool center = false){
    const unsigned long size = data.size();
    const double mean = CommonFnc::computeMean(data);
    const double stddev = CommonFnc::computeStddev(data, mean);
    T min = *(min_element(data.begin(), data.end()));
    T max = *(max_element(data.begin(), data.end()));
    double binSize = (max-min)/(double)bins;

    // Center around mean, so mean is in the middle of the bin.
    if (center){
        const double meanBinLow = mean-binSize/2;
        const double binsMeanLeft = ceil((meanBinLow - min) / binSize);
        const double newMin = meanBinLow - binsMeanLeft * binSize;
        assert(newMin <= min);
        min = newMin;
    }

    // TODO: Simple normality testing on data.
    // Randomize the array
//    std::random_shuffle ( data.begin(), data.end() );

    // T-test, testing u0=u for N(u, q) for unknown q.
    double t0 = (mean/stddev) * sqrt((double)size);
#ifdef BOOST
    students_t distStudent(size - 1);
    double tCrit95 = quantile(complement(distStudent, 0.05 / 2));
    double tCrit99 = quantile(complement(distStudent, 0.01 / 2));
    double tCrit999 = quantile(complement(distStudent, 0.001 / 2));
#else
    double tCrit95 = 1e50, tCrit99 = 1e50, tCrit999 = 1e50;
#endif

    // U-test, testing u0=u for N(u, q) for known q. For us it is 1.
    double u0 = (mean/1) * sqrt((double)size);

    printf("(hist size: %05lu, binSize: %0.6f, min: %0.6f, mean: %0.6f, max: %0.6f, stddev: %0.6f)\n",
           size, binSize, min, mean, max, stddev);

    printf("(hist t0: %0.6f, 95%% reject: %d, 99%% reject: %d, 99.9%% reject: %d)\n",
           t0, abs(t0)>=tCrit95,  abs(t0)>=tCrit99,  abs(t0)>=tCrit999);

    printf("(hist u0: %0.6f, 95%% reject: %d, 99%% reject: %d, 99.9%% reject: %d)\n",
           u0, abs(u0)>=CommonFnc::ucrit(0.05/2), abs(u0)>=CommonFnc::ucrit(0.01/2), abs(u0)>=CommonFnc::ucrit(0.001/2));

    // Very simple test - how many numbers lies in mean - (1.96 x stddev) and mean + (1.96 x stddev).
    u64 liesIn95 = 0;
    u64 liesIn99 = 0;
    double crit95Lo = mean - (CommonFnc::ucrit(0.05/2) * stddev);
    double crit95Hi = mean + (CommonFnc::ucrit(0.05/2) * stddev);
    double crit99Lo = mean - (CommonFnc::ucrit(0.01/2) * stddev);
    double crit99Hi = mean + (CommonFnc::ucrit(0.01/2) * stddev);

    // Binning
    vector<unsigned long> binVector(bins+2);
    fill(binVector.begin(), binVector.end(), 0);

    for(int i = 0; i<size; i++){
        binVector[ (data[i] - min)/binSize ] += 1;

        if (crit95Lo < data[i] &&  data[i] < crit95Hi){
            liesIn95 += 1;
        }

        if (crit99Lo < data[i] &&  data[i] < crit99Hi){
            liesIn99 += 1;
        }
    }

    printf("  #ofValues in 95%% if normal: %llu it is: %0.6f%% of values\n", liesIn95, 100.0*liesIn95/size);
    printf("  #ofValues in 99%% if normal: %llu it is: %0.6f%% of values\n", liesIn99, 100.0*liesIn99/size);

    // Draw the histogram.
    unsigned long binMax = *(max_element(binVector.begin(), binVector.end()));
    for(int i = 0; i < bins; i++){
        printf("%04d[c:%+0.6f]: ", i, min + binSize*i + binSize/2.0);
        int chars = (int)ceil(100 * (binVector[i] / (double)binMax));
        for(int j=0; j<chars; j++){
            cout << "+";
        }
        cout << endl;
    }
}

/**
 * Compute
 */
int testBi(DataSource * dataSource){
    // Number of tests running
    const int numIndependentTests = 200;

    // Should disjoint terms should be used? If yes, for 3order it is 128/3=42 terms.
    const bool disjointTerms = false;

    // Test vector configuration. How many times one term is evaluated.
    const int numTVs = 1024*64; // keep this number divisible by 128 pls!

    // Number of iterations of evaluation.
    const int numEpochs = 1;

    // Number of bytes needed to load from input source for one epoch.
    const unsigned int numBytes = numTVs * TERM_WIDTH_BYTES;
    const u64 inputData2ReadInTotal = ((u64)numIndependentTests)*numTVs*numEpochs*TERM_WIDTH_BYTES;

    // We need to check input file for required size so test has correct interpretation and code is valid.
    long long dataInputSize = dataSource->getAvailableData();
    if (dataInputSize < 0){
        cerr << "Invalid file: " << dataSource->desc() << endl;
        return -1;

    } else if (dataInputSize < inputData2ReadInTotal){
        cerr << "Input file: " << dataSource->desc() << " is too short. Size: " << dataInputSize << " B, required: " << inputData2ReadInTotal << " B" << endl;
        return -2;
    }

    cout << "Using data source: " << dataSource->desc() << ", size: " << setw(4) << (dataInputSize /1024/1024) << " MB, required:"
         << (inputData2ReadInTotal/1024/1024) << " MB" << endl;

    // Allocation, initialization.
    // number of terms we evaluate (affected by degree, disjoint generation strategy, ...)
    u64 termCnt = 0;
    // Buffer for testvector data read from input.
    u8 *TVs = new u8[numBytes];
    vector < bitarray < u64 > * > resultArrays;
    SimpleTerm <u64, u64> s[TERM_WIDTH];
    // Mapping termId -> number of evaluations to 1 for 1 test (all epochs).
    vector<u64> resultStats(TERM_NUMBER);

    // Collecting statistics over all test numbers.
    vector<double> overallFailed95(numIndependentTests);
    vector<double> overallFailed99(numIndependentTests);

    // s[j] = term with only j variable set. Allocate result arrays.
    for (int j = 0; j < TERM_WIDTH; ++j) {
        s[j].alloc(TERM_WIDTH);
        s[j].set(j);
        s[j].allocResults(numTVs);
        resultArrays.push_back(&s[j].getResults());
    }

    for(int testNumber = 0; testNumber < numIndependentTests; ++testNumber) {
        printf("##test: %d/%d\n", testNumber + 1, numIndependentTests);

        // Remembers all results for all polynomials.
        // unordered_map was here before, but we don't need it for now as
        // order on polynomials is well defined for given order - by the generator.
        fill(resultStats.begin(), resultStats.end(), 0);

        // Epoch is some kind of redefined here.
        // Epoch = next processing step of the input data of size numBytes.
        // Statistical processing is done over all epochs. It makes some kind of trade-off between CPU/RAM.
        // If desired, wrap this with another level of loop.
        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            if (numIndependentTests == 1) printf("## EPOCH: %02d\n", epoch);

            // Read test vectors.
            dataSource->read((char *) TVs, numBytes);

            // Single-var term s_j (hw(s_j)=1) is evaluated numTVs times on 128 input bits
            for (int j = 0; j < TERM_WIDTH; ++j) {
                s[j].evaluateTVs(TERM_WIDTH_BYTES, TVs);
            }
            if (numIndependentTests == 1) printf("  elementary results computed\n");

            // Generate all polynomials from precomputed values.
            termRep indices;
            u64 termIdx = 0;
            init_comb(indices, TERM_DEG);
            do {
                // Number of times the polynomial <indices> returned 1 on 128bit test vector.
                int hw = HW_AND<TERM_DEG>(resultArrays, indices);
                resultStats[termIdx++] += (u64) hw;

            } while (next_combination(indices, TERM_WIDTH, disjointTerms));
            if (termCnt == 0) termCnt = termIdx;
        }

        // Result processing.
        const double expectedOccurrences = (numTVs * numEpochs) / (double) (1 << TERM_DEG);
        const double expectedProb = 1.0 / (1 << TERM_DEG);
        if (numIndependentTests == 1) printf("Expected occ: %.6f, expected prob: %.6f\n", expectedOccurrences, expectedProb);

#ifdef DUMP_FILES
        ofstream scoreFile("./zscores.csv", ios::trunc);
        ofstream polyFile("./polynomials.csv", ios::trunc);
        //scoreFile << "polyIdx;zscore" << endl;
#endif

        vector<double> zscores(termCnt);
        u64 polyTotalCtr = 0;
        u64 totalObserved = 0;
        u64 rejected95 = 0;
        u64 rejected99 = 0;
        double zscoreTotal = 0;
        termRep indices;
        init_comb(indices, TERM_DEG);
        do {
            u64 observed = resultStats[polyTotalCtr];
            totalObserved += observed;

            double observedProb = (double) observed / numTVs / numEpochs;
            double zscore = CommonFnc::zscore(observedProb, expectedProb, numTVs * numEpochs);
            double zscoreAbs = abs(zscore);
            if (zscoreAbs >= CommonFnc::ucrit(0.05 / 2)) {
                rejected95 += 1;
            }
            if (zscoreAbs >= CommonFnc::ucrit(0.01 / 2)) {
                rejected99 += 1;
            }

            zscores[polyTotalCtr] = zscore;
            zscoreTotal += zscoreAbs;
            if (numIndependentTests == 1 && polyTotalCtr < 128) {
                printf("Observed[%08x]: %08llu, probability: %.6f, z-score: %0.6f\n",
                       (unsigned) polyTotalCtr, observed, observedProb, zscore);
            }

#ifdef DUMP_FILES
            //scoreFile << polyTotalCtr << ";" << zscore << endl;
            scoreFile << zscore << endl;
            polyFile << observed << endl;
#endif

            polyTotalCtr += 1;
        } while (next_combination(indices, TERM_WIDTH, disjointTerms));

#ifdef DUMP_FILES
        scoreFile.close();
        polyFile.close();
#endif

        // Store all data over all tests.
        overallFailed95[testNumber] = (double)rejected95 / polyTotalCtr;
        overallFailed99[testNumber] = (double)rejected99 / polyTotalCtr;

        // Info dumping phase, do only for the last experiment.
        if (testNumber+1 != numIndependentTests){
            continue;
        }

        printf("Dumping statistics from the last test:\n");
        double avgOcc = (double) totalObserved / polyTotalCtr;
        double avgProb = avgOcc / (numTVs * numEpochs);

        // TODO: normality test for zscores.
        printf("z-score histogram: \n");
        histogram(zscores, 31, true);

        printf("      totalTerms: %04llu, acc: %08llu, average occurrence: %0.6f, average prob: %0.6f\n",
               polyTotalCtr, totalObserved, avgOcc, avgProb);

        printf("      ztotal: %0.6f, avg-zscore: %0.6f\n", zscoreTotal, zscoreTotal / polyTotalCtr);
        printf("      data processed: %0.2f kB = %0.2f MB\n",
               inputData2ReadInTotal / 1024.0,
               inputData2ReadInTotal / 1024.0 / 1024);

        // Test Bi(polyTotalCtr, 0.05), Bi(polyTotalCtr, 0.01).
        printf("# of rejected 95%%: %04llu that is %0.6f%%, zscore: %0.6f\n",
               rejected95, 100.0 * rejected95 / polyTotalCtr,
               CommonFnc::zscore((double) rejected95 / polyTotalCtr, 0.05, polyTotalCtr));

        printf("# of rejected 99%%: %04llu that is %0.6f%%, zscore: %0.6f\n",
               rejected99, 100.0 * rejected99 / polyTotalCtr,
               CommonFnc::zscore((double) rejected99 / polyTotalCtr, 0.01, polyTotalCtr));
    }

    // Test statistics
    printf("--------------------------------------------------------------------\n");
    printf("---- Testing completed --- \n");
    printf("--------------------------------------------------------------------\n");

    printf("\nHistogram for ratio of failed hypotheses with alpha=0.05:\n");
    histogram(overallFailed95, 21, true);
    // TODO: T-test for mean = 0.05


    printf("\nHistogram for ratio of failed hypotheses with alpha=0.01:\n");
    histogram(overallFailed99, 21, true);
    // TODO: T-test for mean = 0.01


    return 0;
}



int main(int argc, char *argv[]) {
    initState();
    unsigned long seed = (unsigned long) random();
    printf("Main seed number: %lu\n", seed);

    std::unique_ptr<DataSourceFile>         dsFile(nullptr);
    std::unique_ptr<DataSourceAES>          dsAES(new DataSourceAES(seed));
    std::unique_ptr<DataSourceMD5>          dsMD5(new DataSourceMD5(seed));
    std::unique_ptr<DataSourceRC4>          dsRC4(new DataSourceRC4(seed));
    std::unique_ptr<DataSourceRC4Column>    dsRC4Col(new DataSourceRC4Column(seed, TERM_WIDTH_BYTES));
    std::unique_ptr<DataSourceSystemRandom> dsSys(new DataSourceSystemRandom(seed));
    DataSource * dsToUse = dsAES.get();

    if (argc >= 2){
        std::string fileName = argv[1];
        dsFile.reset(new DataSourceFile(fileName));
        dsToUse = dsFile.get();
    }

    // Try AES for now
    dsToUse = dsAES.get();

    // Test - watch out which data source is passed in!
    testBi(dsToUse);

    return 0;
}