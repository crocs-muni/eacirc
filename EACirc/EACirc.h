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

#define EACIRC_CONFIG_LOADED 0x01
#define EACIRC_PREPARED 0x02
#define EACIRC_INITIALIZED 0x04

class EACirc {
    //! current error status
    int m_status;
    //! is evolution on? TODO: move to main settings?
    bool m_evolutionOff;
    //! should genome be loaded at the beginning? TODO: move to main settings?
    bool m_loadGenome;
    //! main seed for this computation (can be reproduced with this seed and settings)
    unsigned long m_originalSeed;
    //! seed used to reseed GAlib the last time
    unsigned long m_currentGalibSeed;
    //! structure of main settings
    BASIC_INIT_DATA basicSettings;
    //! evaluaor
    Evaluator* m_evaluator;
    //! genetics algorithm instance
    GASteadyStateGA* m_gaData;
    //! state of preparations we are in (config loaded? state initialized? see defines)
    unsigned int m_readyToRun;
    //! generation currently computed
    int m_actGener;

    /** reinstantialized genetic algorithm object and resets population stats
      * - GAlib is reseeded by current seed at the beginning
      * @param population   will be used to init new instance of genetic algorithm
      *                     (given by reference to ease computation)
      */
    void seedAndResetGAlib(const GAPopulation& population);

    /** saves current state (random generators, seeds, etc.) to file
      * - random generator states
      * - main seed, current galib seed
      * - number of generations required and finished
      * @param filename     file to write to (contents overwritten)
      * @return status
      */
    int saveState(const string filename) const;

    /** loads state from file
      * - creates random generators
      * - restores main seed and currnet galib seed
      * - GAlib is not yet reseeded!
      * @param filename     file to read from
      */
    void loadState(const string filename);

    /** create new initial state
      * - generate main seed if needed
      * - create all random generatos needed
      * - seed GAlib with newly generated seed
      */
    void createState();

    /** saves current population to file
      * - population size
      * - genome size
      * - full genome for each individual in a population
      * @param filename     file to write to (contents overwritten)
      * @return status
      */
    int savePopulation(const string filename) const;

    /** load population from file
      * full genome is loaded fro each saved individual
      * - GAlib seeded and reset
      * @param filename     file to read from (binary population)
      */
    void loadPopulation(const string filename);

    /** creates new initial population according to settings
      * - seeds galib
      * - load single genome if needed
      * TODO: change to loading population of size 1?
      * - reseed and reset GAlib with newly generated seed
      */
    void createPopulation();
public:
    EACirc();
    EACirc(bool evolutionOff);
    ~EACirc();

    /** loads configuration from xml file to settings attribute
      * checks basic consistency of the settings
      * @param filename     configuration file to use
      */
    void loadConfiguration(const string filename);

    /** according to settings either loads state and population
      * or creates new initial state and population
      * must be called after loading settings
      */
    void initializeState();

    /** does the necessary peparations needed just before running
      * must be called after initializing state (either loading or creating)
      */
    void prepare();

    /** runs the main program loop accoring to settings
      * must be run after preparing
      */
    void run();

    /** returns current error status
      * @return status
      */
    int getStatus() const;

    /** save computation (current state and population) to allow recommencing later
      * @param stateFilename        file to save the state (contents overwritten)
      * @param populationFilename   file to save the population (contents overwritten)
      */
    int saveProgress(const string stateFilename, const string populationFilename) const;
};

#endif
