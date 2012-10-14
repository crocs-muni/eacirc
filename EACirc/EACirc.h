#ifndef EACIRC_H
#define EACIRC_H

#include "EACglobals.h"
#include "random_generator/RndGen.h"
//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
#include "estream/EncryptorDecryptor.h"

extern IRndGen* rndGen;
extern IRndGen* biasRndGen;
extern GA_CIRCUIT* pGACirc;
extern EncryptorDecryptor* encryptorDecryptor;

//CEACircuit();
//~CEACircuit();
int main(int argc, char **argv);
//static UINT RunGACircuit(LPVOID pParam);
//static int ExecuteFromText(string textCircuit, GA1DArrayGenome<unsigned long> *genome); 
static int LoadConfigScript(string filePath, BASIC_INIT_DATA* pBasicSettings);

#endif
