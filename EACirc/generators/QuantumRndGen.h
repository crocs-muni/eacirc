#ifndef QUANTUMRNDGEN_H
#define QUANTUMRNDGEN_H

#include <string>
#include "generators/IRndGen.h"

class QuantumRndGen : public IRndGen{
private:
    //! temporary data (accumulator)
    unsigned char* m_accumulator;
    //! using rng output files?
    bool m_usesQRNGData;
    //! path to rng output files
    string m_QRNGDataPath;
    //! real data length in accumulator
    int m_accLength;
    //! current accumulator position
    int m_accPosition;
    //! which QRNG file is currently used?
    int m_fileIndex;
    //! internal MD5 generator if quantum files not found
    IRndGen* m_internalRNG;
    //! get filename of random file with index with path
    string getQRNGDataFileName(int fileIndex);
    //! check whether qrng data are available and set attributes accordingly
    void checkQRNGdataAvailability();
public:
    QuantumRndGen(unsigned long seed = 0, string QRBGSPath = "");
    QuantumRndGen(TiXmlNode* pRoot);
    // QuantumRndGen(const QuantumRndGen&) = delete; //(not supprrted in MS VS)
    // const QuantumRndGen& operator =(const QuantumRndGen&) = delete; //(not supprrted in MS VS)
    ~QuantumRndGen();

    int getRandomFromInterval(unsigned long highBound, unsigned long *pRandom);
    int getRandomFromInterval(unsigned char highBound, unsigned char *pRandom);
    int getRandomFromInterval(unsigned int highBound, unsigned int *pRandom);
    int getRandomFromInterval(int highBound, int *pRandom);
    int getRandomFromInterval(float highBound, float *pRandom);
    int discartValue();

    string shortDescription() const;
    // implemented in XMLProcessor:
    TiXmlNode* exportGenerator() const;
protected:
    int updateAccumulator();
    int loadQRNGDataFile();
    int reinitRandomGenerator();
};

#endif // QUANTUMRNDGEN_H
