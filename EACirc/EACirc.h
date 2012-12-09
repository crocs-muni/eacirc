#ifndef EACIRC_H
#define EACIRC_H

#include "EACglobals.h"
#include "random_generator/IRndGen.h"
//#include "random_generator/QuantumRndGen.h"
//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
#include "Evaluator.h"
#include "estream/EncryptorDecryptor.h"

extern IRndGen* rndGen;
extern IRndGen* biasRndGen;
extern GA_CIRCUIT* pGACirc;
extern EncryptorDecryptor* encryptorDecryptor;

class EACirc {

#define EACIRC_CONFIG_LOADED 0x01
#define EACIRC_PREPARED 0x02
#define EACIRC_INITIALIZED 0x04

    int m_status;
    bool m_evolutionOff;
    bool m_loadGenome;
    unsigned long m_seed;
    BASIC_INIT_DATA pBasicSettings;
    Evaluator* m_evaluator;
    unsigned int m_readyToRun;
public:
    EACirc();
    EACirc(bool evolutionOff);
    ~EACirc();
    void loadConfiguration(string filename);
    void loadState(string filename);
    void saveState(string filename);
    void initializeState();
    void prepare();
    void run();
    int getStatus();
};

int main(int argc, char **argv);

//static UINT RunGACircuit(LPVOID pParam);
//static int ExecuteFromText(string textCircuit, GA1DArrayGenome<unsigned long> *genome); 

#endif
