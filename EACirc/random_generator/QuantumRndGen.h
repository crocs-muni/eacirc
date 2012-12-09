#ifndef QUANTUMRNDGEN_H
#define QUANTUMRNDGEN_H

#include <string>
#include "random_generator/IRndGen.h"

class QuantumRndGen : public IRndGen{
    unsigned char* m_accumulator;
    bool m_usesQRNGData; // using rng output files?
    string m_QRNGDataPath; // path to rng output files
    int m_accLength; // real data length
    int m_accPosition; // accumulator position
    int m_fileIndex; // which QRNG file is currently used?
    // minstd_rand m_internalRNG;
    IRndGen* m_internalRNG;
public:
    QuantumRndGen(unsigned long seed = 0, string QRBGSPath = "");
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

#endif // QUANTUMRNDGEN_H
