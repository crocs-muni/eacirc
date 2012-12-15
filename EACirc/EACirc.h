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
    unsigned long m_originalSeed;
    unsigned long m_currentGalibSeed;
    BASIC_INIT_DATA basicSettings;
    Evaluator* m_evaluator;
    GASteadyStateGA* m_gaData;
    unsigned int m_readyToRun;
    int m_actGener;
    void loadInitialState(string stateFilename, string populationFilename);
    void createNewInitialState();
    void seedAndResetGAlib(GAPopulation population);
public:
    EACirc();
    EACirc(bool evolutionOff);
    ~EACirc();
    void loadConfiguration(string filename);
    void saveState(string stateFilename, string populationFilename);
    void initializeState();
    void prepare();
    void run();
    int getStatus();
};

//static UINT RunGACircuit(LPVOID pParam);
//static int ExecuteFromText(string textCircuit, GA1DArrayGenome<unsigned long> *genome); 

#endif
