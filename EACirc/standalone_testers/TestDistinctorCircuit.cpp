#include "standalone_testers/TestDistinctorCircuit.h"
#include "EACglobals.h"
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
//#include "globals.h"
#include "EACirc.h"
#include "CommonFnc.h"
#include "random_generator/IRndGen.h"
#include "random_generator/BiasRndGen.h"
#include "GA1DArrayGenome.h"
#include "tinyxml.h"
#include "math.h"
#include "time.h"
#include "CircuitGenome.h"
#include "projects/estream/estreamInterface.h"
#include "projects/estream/EncryptorDecryptor.h"
#include "EAC_circuit.h"

//TEST THE OUTPUT FILE AGAINT TEST DATA
int testDistinctorCircuit(string filename1, string filename2) {
	ifstream		file;
	ifstream		file2;
	int			length;
	int			length2;
	unsigned char inputs[MAX_INPUTS];
	unsigned char outputs[MAX_OUTPUTS];
	unsigned char *accumulator = new unsigned char[40000000];
	unsigned char *accumulator2 = new unsigned char[40000000];

    file.open(filename1, fstream::in | fstream::binary);
    file2.open(filename2, fstream::in | fstream::binary);

    if (!file.is_open() || !file2.is_open()) {
        mainLogger.out() << "error: Could not find/open file." << endl;
        mainLogger.out() << "       Required files: " << filename1.c_str() << ", " << filename2.c_str() << endl;
        return STAT_FILE_OPEN_FAIL;
    }

    string resultsFile = "Distinctor_results.txt";
    ofstream results(resultsFile.c_str(), ios::app);
    results << "---------- results for distinctor circuit ----------" << endl;

    file.seekg (0, ios::end);
    file2.seekg (0, ios::end);
    length = file.tellg();
    length2 = file2.tellg();
    file.seekg (0, ios::beg);
    file2.seekg (0, ios::beg);
    if (length > 40000000) length = 40000000;
    if (length2 > 40000000) length2 = 40000000;
    file.read((char*)accumulator, length);
    file2.read((char*)accumulator2, length2);
    int i = 0;
    int match = 0;
    int predictors = 0;
    results << "=== Testset 1 ===" << endl;
    for (i=0; i<length;i+=pGACirc->settings->testVectors.testVectorLength) {
        for (int e=0; e<pGACirc->settings->testVectors.testVectorLength; e++) {
            inputs[e] = accumulator[i+e];
        }
        circuit(inputs, outputs);
        for (int e=0; e<pGACirc->settings->circuit.sizeOutputLayer; e++){
            if (outputs[e] <= UCHAR_MAX/2) {
                match++;
            }
            predictors++;
        }
        if (i > 0 && (i%(pGACirc->settings->testVectors.numTestVectors*pGACirc->settings->testVectors.testVectorLength) == 0)) {
            results << "Match: " << match << ", Predictors: " << predictors << ", Fitness: " << float(match)/float(predictors)<< endl;
            match = 0;
            predictors = 0;
        }
    }

    match = 0;
    predictors = 0;
    results << " === Testset 2 ===" << endl;
    for (i=0; i<length2;i+=pGACirc->settings->testVectors.testVectorLength) {
        for (int e=0; e<pGACirc->settings->testVectors.testVectorLength; e++){
            inputs[e] = accumulator2[i+e];
        }
        circuit(inputs, outputs);
        for (int e=0; e<pGACirc->settings->circuit.sizeOutputLayer; e++) {
            if (outputs[e] > UCHAR_MAX/2) {
                match++;
            }
            predictors++;
        }
        if (i > 0 && (i%(pGACirc->settings->testVectors.numTestVectors*pGACirc->settings->testVectors.testVectorLength) == 0)) {
            results << "Match: " << match << ", Predictors: " << predictors << ", Fitness: " << float(match)/float(predictors)<< endl;
            match = 0;
            predictors = 0;
        }
    }

	delete[] accumulator;
	delete[] accumulator2;
    results.close();

    return STAT_OK;
}
