//#include "globals.h"
#include "EACirc.h"
#include "CommonFnc.h"
//libinclude (galib/GA1DArrayGenome.h)
#include "GA1DArrayGenome.h"
#include "test_vector_generator/EstreamTestVectGener.h"
#include "estream/EncryptorDecryptor.h"
#include "estream/estreamInterface.h"

EstreamTestVectGener::EstreamTestVectGener() : ITestVectGener() {
	this->numstats = new int[2];
	Init();
}

EstreamTestVectGener::~EstreamTestVectGener() {
    delete[] this->numstats;
}

void EstreamTestVectGener::Init() {	

}

void EstreamTestVectGener::getTestVector(){

    ofstream tvfile(FILE_TEST_VECTORS, ios::app);

	int streamnum = 0;
	bool error = false;

	u8 plain[MAX_INPUTS];// = new u8[STREAM_BLOCK_SIZE];
	u8 outplain[MAX_INPUTS];// = new u8[STREAM_BLOCK_SIZE];

	switch (pGACirc->testVectorEstreamMethod) {
		case TESTVECT_ESTREAM_DISTINCT:

			//SHALL WE BALANCE TEST VECTORS?
			if (pGACirc->testVectorBalance && (numstats[0] >= pGACirc->numTestVectors/2))
				streamnum = 1;
			else if (pGACirc->testVectorBalance && (numstats[1] >= pGACirc->numTestVectors/2))
				streamnum = 0;
			else
				rndGen->getRandomFromInterval(1, &streamnum);
			numstats[streamnum]++;
			//Signalize the correct value
			for (int output = 0; output < pGACirc->numOutputs; output++) outputs[output] = streamnum * 0xff;

			//generate the plaintext for stream
            if ((streamnum == 0 && pGACirc->testVectorEstream != ESTREAM_RANDOM) ||
                (streamnum == 1 && pGACirc->testVectorEstream2 != ESTREAM_RANDOM) ) {
				if (pGACirc->saveTestVectors == 1)
					tvfile  << "(alg n." << ((streamnum==0)?pGACirc->testVectorEstream:pGACirc->testVectorEstream2) << " - " << ((streamnum==0)?pGACirc->limitAlgRoundsCount:pGACirc->limitAlgRoundsCount2) << " rounds): ";

				if (pGACirc->estreamInputType == ESTREAM_GENTYPE_ZEROS)
					for (int input = 0; input < pGACirc->testVectorLength; input++) plain[input] = 0x00;
				else if (pGACirc->estreamInputType == ESTREAM_GENTYPE_ONES)
					for (int input = 0; input < pGACirc->testVectorLength; input++) plain[input] = 0x01;
				else if (pGACirc->estreamInputType == ESTREAM_GENTYPE_RANDOM)
					for (int input = 0; input < pGACirc->testVectorLength; input++) rndGen->getRandomFromInterval(255, &(plain[input]));
				else if (pGACirc->estreamInputType == ESTREAM_GENTYPE_BIASRANDOM)
					for (int input = 0; input < pGACirc->testVectorLength; input++) biasRndGen->getRandomFromInterval(255, &(plain[input]));
	
				encryptorDecryptor->encrypt(plain,inputs,streamnum);
				encryptorDecryptor->decrypt(inputs,outplain,streamnum+2);
				
			}
			else { // RANDOM
				if (pGACirc->saveTestVectors == 1)
                    tvfile << "(RANDOM INPUT - " << rndGen->shortDescription() << "):";
				for (int input = 0; input < pGACirc->testVectorLength; input++) {
					rndGen->getRandomFromInterval(255, &outplain[input]);
					plain[input] = inputs[input] = outplain[input];
				}
			}

			for (int input = 0; input < pGACirc->testVectorLength; input++) {
				if (outplain[input] != plain[input]) {
                    ofstream fitfile(FILE_FITNESS_PROGRESS, ios::app);
                    fitfile << "Error! Decrypted text doesn't match the input. See " << FILE_TEST_VECTORS << " for details." << endl;
					fitfile.close();

					// SIGNALIZE THE ERROR - WE NEED TO LOG INPUTS/OUTPUTS
					error = true;
					//exit(1);
					break;
				}
			}

			break;

	case TESTVECT_ESTREAM_BITS_TO_CHANGE:
		//generate the plaintext
			if (pGACirc->estreamInputType == ESTREAM_GENTYPE_ZEROS)
				for (int input = 0; input < pGACirc->testVectorLength; input++) inputs[input] = 0x00;
			else if (pGACirc->estreamInputType == ESTREAM_GENTYPE_ONES)
				for (int input = 0; input < pGACirc->testVectorLength; input++) inputs[input] = 0x01;
			else if (pGACirc->estreamInputType == ESTREAM_GENTYPE_RANDOM)
				for (int input = 0; input < pGACirc->testVectorLength; input++) rndGen->getRandomFromInterval(255, &(inputs[input]));
			else if (pGACirc->estreamInputType == ESTREAM_GENTYPE_BIASRANDOM)
				for (int input = 0; input < pGACirc->testVectorLength; input++) biasRndGen->getRandomFromInterval(255, &(inputs[input]));

			// WE NEED TO LET EVALUATOR KNOW THE INPUTS
			for (int input = 0; input < pGACirc->testVectorLength; input++)
				outputs[input] = inputs[input];

		break;

		default:
			assert(false);
			break;
	}
	
	// SAVE TEST VECTORS IN BINARY FILES
	if (pGACirc->saveTestVectors == 1) {
		if (streamnum == 0) {
            ofstream itvfile(FILE_TEST_DATA_1, ios::app | ios::binary);
			for (int input = 0; input < pGACirc->testVectorLength; input++) {
					itvfile << inputs[input];
			}
			itvfile.close();
		}
		else {
            ofstream itvfile(FILE_TEST_DATA_2, ios::app | ios::binary);
			for (int input = 0; input < pGACirc->testVectorLength; input++) {
					itvfile << inputs[input];
			}
			itvfile.close();
		}

		int tvg = 0;
		if (streamnum == 0) tvg = pGACirc->testVectorEstream;
		else tvg = pGACirc->testVectorEstream2;
		tvfile << setfill('0');

		if (memcmp(inputs,plain,pGACirc->testVectorLength) != 0) {
			for (int input = 0; input < pGACirc->testVectorLength; input++)
			tvfile << setw(2) << hex << (int)(plain[input]);
			tvfile << "::";
		}

		for (int input = 0; input < pGACirc->testVectorLength; input++)
			tvfile << setw(2) << hex << (int)(inputs[input]);

		if (memcmp(inputs,outplain,pGACirc->testVectorLength) != 0) {
			tvfile << "::";
			for (int input = 0; input < pGACirc->testVectorLength; input++)
			tvfile << setw(2) << hex << (int)(outplain[input]);
		}
		tvfile << endl;
	}

	tvfile.close();
	
	// THERE WAS AN ERROR, EXIT...
	if (error) exit(1);

	return;
}

void EstreamTestVectGener::generateTestVectors() {
	// USED FOR BALANCING TEST VECTORS
	this->numstats[0] = 0;
	this->numstats[1] = 0;

	for (int testSet = 0; testSet < pGACirc->numTestVectors; testSet++) {
		if (pGACirc->saveTestVectors == 1) {
            ofstream tvfile(FILE_TEST_VECTORS, ios::app);
			tvfile << "Testset n." << dec << testSet << endl;
			tvfile.close();
		}

		getTestVector();
		for (int input = 0; input < MAX_INPUTS; input++) {
			pGACirc->testVectors[testSet][input] = inputs[input];
		}
		for (int output = 0; output < pGACirc->numOutputs; output++) 
			pGACirc->testVectors[testSet][MAX_INPUTS+output] = outputs[output];
	}
}
