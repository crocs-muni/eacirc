#include "../SSGlobals.h"
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../globals.h"
#include "../EACirc.h"
#include "../CommonFnc.h"
#include "../status.h"
#include "../Random Generator/IRndGen.h"
#include "../Random Generator/BiasRndGen.h"
#include "../ga/GA1DArrayGenome.h"
#include "../tinyXML/tinyxml.h"
#include "math.h"
#include "time.h"
#include "../Evaluator.h"
#include "../CircuitGenome.h"
#include "../estream-interface.h"
#include "../ITestVectGener.h"
#include "../EstreamVectGener.h"
#include "../EncryptorDecryptor.h"
#include "../EAC_circuit.h"

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
	if (file.is_open() && file2.is_open()) {
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
		cout << "===Testset 1===" << endl;
		for (i=0; i<length;i+=pGACirc->testVectorLength) {
			for (int e=0; e<pGACirc->testVectorLength; e++) {
				inputs[e] = accumulator[i+e];
			}
			circuit(inputs, outputs);
			for (int e=0; e<pGACirc->outputLayerSize; e++){
				if (outputs[e] <= UCHAR_MAX/2) {
					match++;
				}
				predictors++;
			}
			if (i > 0 && (i%(pGACirc->numTestVectors*pGACirc->testVectorLength) == 0)) {
				cout << "Match: " << match << ", Predictors: " << predictors << ", Fitness: " << float(match)/float(predictors)<< endl;
				match = 0;
				predictors = 0;
			}
		}

		match = 0;
		predictors = 0;
		cout << "===Testset 2===" << endl;
		for (i=0; i<length2;i+=pGACirc->testVectorLength) {
			for (int e=0; e<pGACirc->testVectorLength; e++){
				inputs[e] = accumulator2[i+e];
			}
			circuit(inputs, outputs);
			for (int e=0; e<pGACirc->outputLayerSize; e++) {
				if (outputs[e] > UCHAR_MAX/2) {
					match++;
				}
				predictors++;
			}
			if (i > 0 && (i%(pGACirc->numTestVectors*pGACirc->testVectorLength) == 0)) {
				cout << "Match: " << match << ", Predictors: " << predictors << ", Fitness: " << float(match)/float(predictors)<< endl;
				match = 0;
				predictors = 0;
			}
		}
	}
	delete[] accumulator;
	delete[] accumulator2;
	return 0;
}
