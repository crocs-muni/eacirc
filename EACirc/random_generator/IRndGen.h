#ifndef RANDOM_GENERATOR_INTERFACE_H
#define RANDOM_GENERATOR_INTERFACE_H

//#include "EACglobals.h"
#include <string>
#include <random>
#include "XMLProcessor.h"
//#include "MD5RndGen.h"

class IRndGen {
//private:
//    static MD5RndGen* m_mainGenerator;
protected:
    int m_type; // generator type, see EACirc constants
    unsigned long m_seed; // original seed
public:
    IRndGen(int type, unsigned long seed);
    virtual ~IRndGen() {}

    //static void initMainGenerator(unsigned long seed);
    //static void initMainGenerator(TiXmlNode* pRoot);
    //static unsigned long getRandomFromMainGenerator();
    //static TiXmlNode* exportMainGenerator();

    virtual int getRandomFromInterval(unsigned long, unsigned long *) = 0;
    virtual int getRandomFromInterval(unsigned char, unsigned char *) = 0;
    virtual int getRandomFromInterval(int, int *) = 0;
    virtual int getRandomFromInterval(float, float *) = 0;
    virtual int discartValue() = 0;
    virtual int reinitRandomGenerator() = 0;

    virtual string shortDescription() const = 0;
    virtual TiXmlNode* exportGenerator() const = 0;
};

ostream& operator <<(ostream& out, const IRndGen& generator);
istream& operator >>(istream& in, IRndGen& generator);

#endif
