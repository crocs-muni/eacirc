#ifndef MD5RNDGEN_H
#define MD5RNDGEN_H

#include "md5.h"
#include "IRndGen.h"

class MD5RndGen : public IRndGen {
    MD5_DIGEST m_md5Accumulator; // accumulator for MD5
public:
    MD5RndGen(unsigned long seed = 0);
    MD5RndGen(TiXmlNode* pRoot);
    // MD5RndGen(const MD5RndGen&) = delete; //(not supprrted in MS VS)
    // const MD5RndGen& operator =(const MD5RndGen&) = delete; //(not supprrted in MS VS)
    ~MD5RndGen() {}

    int getRandomFromInterval(unsigned long, unsigned long *);
    int getRandomFromInterval(unsigned char, unsigned char *);
    int getRandomFromInterval(unsigned int, unsigned int *);
    int getRandomFromInterval(int, int *);
    int getRandomFromInterval(float, float *);
    int discartValue();

    string shortDescription() const;
    TiXmlNode* exportGenerator() const;
    
protected:
    int updateAccumulator();
};

#endif // MD5RNDGEN_H
