#ifndef RANDOM_GENERATOR_INTERFACE_H
#define RANDOM_GENERATOR_INTERFACE_H

//#include "EACglobals.h"
#include <string>
#include <random>
#include "XMLProcessor.h"
#include "CommonFnc.h"

class IRndGen {
protected:
    //! generator type, see EACirc constants
    int m_type;
    //! original seed used when generator was created
    unsigned long m_seed;
public:
    IRndGen(int type, unsigned long seed);
    virtual ~IRndGen() {}

    static IRndGen* parseGenerator(TiXmlNode* pRoot);

    virtual int getRandomFromInterval(unsigned long, unsigned long *) = 0;
    virtual int getRandomFromInterval(unsigned char, unsigned char *) = 0;
    virtual int getRandomFromInterval(int, int *) = 0;
    virtual int getRandomFromInterval(float, float *) = 0;
    virtual int discartValue() = 0;

    virtual string shortDescription() const = 0;
    virtual TiXmlNode* exportGenerator() const = 0;
};

ostream& operator <<(ostream& out, const IRndGen& generator);
istream& operator >>(istream& in, IRndGen& generator);

#endif
