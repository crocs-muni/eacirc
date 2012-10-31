#ifndef RANDOM_GENERATOR_INTERFACE_H
#define RANDOM_GENERATOR_INTERFACE_H

#include "EACglobals.h"
#include <string>
#include <random>
#include "XMLProcessor.h"

class IRndGen {
private:
    static minstd_rand m_mainGenerator;
protected:
    int m_type;
public:
    IRndGen() {}
    virtual ~IRndGen() {}

    static void initMainGenerator(unsigned long seed) { m_mainGenerator.seed(seed); }
    static unsigned long getRandomFromMainGenerator() { return m_mainGenerator(); }

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
