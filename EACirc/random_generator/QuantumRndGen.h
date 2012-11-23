#include <string>
#include "random_generator/IRndGen.h"

#ifndef RNDGEN_H
#define RNDGEN_H

class QuantumRndGen : public IRndGen{
    unsigned char* m_accumulator;
    bool m_usesQRNGData; // using rng output files?
    string m_QRNGDataPath; // path to rng output files
    int m_accLength; // real data length
    int m_accPosition; // accumulator position
    long m_seed; // seed
    int m_fileIndex; // which QRNG file is currently used?
    minstd_rand m_internalRNG;
public:
    QuantumRndGen(unsigned long m_seed = 0, string QRBGSPath = "");
    // implemented in XMLProcessor:
    QuantumRndGen(TiXmlNode* pRoot);
    // QuantumRndGen(const QuantumRndGen&) = delete; //(not supprrted in MS VS)
    // const QuantumRndGen& operator =(const QuantumRndGen&) = delete; //(not supprrted in MS VS)
    ~QuantumRndGen();

    int getRandomFromInterval(unsigned long highBound, unsigned long *pRandom);
    int getRandomFromInterval(unsigned char highBound, unsigned char *pRandom);
    int getRandomFromInterval(int highBound, int *pRandom);
    int getRandomFromInterval(float highBound, float *pRandom);
    int discartValue();
    int reinitRandomGenerator();

    string shortDescription() const;
    // implemented in XMLProcessor:
    TiXmlNode* exportGenerator() const;
protected:
    int updateAccumulator();
    int loadQRNGDataFile();
};

#endif
