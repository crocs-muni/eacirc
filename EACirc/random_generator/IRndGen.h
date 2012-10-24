#ifndef RANDOM_GENERATOR_INTERFACE_H
#define RANDOM_GENERATOR_INTERFACE_H

#include "EACglobals.h"
#include <string>

class IRndGen {
    int type;
public:
    IRndGen();
    IRndGen(int);
    virtual ~IRndGen() {}
    //IRndGen* getRndGenClass(void);
    IRndGen* getInitializedRndGenClass(unsigned long seed = 0, std::string QRBGSPath = "");
    virtual int GetRandomFromInterval(unsigned long, unsigned long *){return 0;}
    virtual int GetRandomFromInterval(unsigned char, unsigned char *){return 0;}
    virtual int GetRandomFromInterval(int, int *){return 0;}
    virtual int GetRandomFromInterval(float, float *){return 0;}
    virtual int InitRandomGenerator(unsigned long seed = 0, std::string QRBGSPath = ""){return 0;}
    virtual string ToString(){return "";}
};

#endif
