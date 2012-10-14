//#include "stdafx.h"
#include "RndGen.h"
#include "time.h"

CRndGen::CRndGen(unsigned long seed, string QRBGSPath) {
	accumulator = NULL;
    InitRandomGenerator(seed, QRBGSPath);
}


int CRndGen::GetRandomFromInterval(unsigned long highBound, unsigned long *pRandom) {
    int     status = STAT_OK;

	if (highBound != ULONG_MAX) highBound++;
    // GET FIRST DWORD FROM ACCUMULATOR     
    unsigned long   random;
    memcpy(&random, accumulator+accPosition, sizeof(unsigned long));
    if (pRandom) {
		*pRandom = (unsigned long) (((float) random / ULONG_MAX) *  highBound);
		if (*pRandom == highBound) *pRandom = 0;
	}
	// UPDATE ACCUMULATOR
    UpdateAccumulator();

    return status;
}

int CRndGen::GetRandomFromInterval(unsigned char highBound, unsigned char *pRandom) {
    int     status = STAT_OK;
    unsigned long   rand = 0;
    
    status = GetRandomFromInterval(highBound, &rand);
    *pRandom = (unsigned char) rand;

    return status;
}

int CRndGen::GetRandomFromInterval(int highBound, int *pRandom) {
    int     status = STAT_OK;

	if (highBound != INT_MAX) highBound++;
    // GET FIRST DWORD FROM ACCUMULATOR     
    int   random;
    memcpy(&random, accumulator+accPosition, sizeof(int));
    // SUPRESS NEGATIVE VALUES
    random = abs(random);
    if (pRandom) {
		*pRandom = (int) (((float) random / INT_MAX) *  highBound);
		if (*pRandom == highBound) *pRandom = 0;
	}

	// UPDATE ACCUMULATOR
    UpdateAccumulator();

    return status;
}

int CRndGen::GetRandomFromInterval(float highBound, float *pRandom) {
    int     status = STAT_OK;

	if (highBound != ULONG_MAX) highBound++;
    // GET FIRST DWORD FROM ACCUMULATOR     
    unsigned long   random;
    memcpy(&random, accumulator+accPosition, sizeof(unsigned long));
    if (pRandom) {
		*pRandom = (float) (((float) random / ULONG_MAX) *  highBound);
		if (*pRandom == highBound) *pRandom = 0;
	}

	// UPDATE ACCUMULATOR
    UpdateAccumulator();

    return status;
}

int CRndGen::InitRandomGenerator(unsigned long seed, string QRBGSPath) {
	
	//PREVENT LEAKS
	if (accumulator != 0) delete[] accumulator;

	this->seed = seed;
	this->bQRGBSPath = QRBGSPath;
    
    // IF SEED NOT EXTERNALLY SUPPLIED, TAKE SYSTEM TIME
    if (seed == 0) seed = (unsigned int) time(NULL);
    
    // INITIALIZE GENERATOR
    srand(seed);
	
	ifstream		file;
    int				fileIndex = rand() % FILE_QRNG_DATA_INDEX_MAX;
	
	// CHECK FOR QUANTUM DATA SOURCE
	ostringstream os1;
    os1 << bQRGBSPath << FILE_QRNG_DATA_PREFIX << fileIndex << FILE_QRNG_DATA_SUFFIX;
	string fileName;
	fileName = os1.str();
	file.open(fileName.c_str(), fstream::in | fstream::binary);
	if (file.is_open()) {
		bQRGBS = true;

		int length;

		file.seekg (0, ios::end);
		length = file.tellg();
		file.seekg (0, ios::beg);

		// READ CONTENT OF FILE
        if (RANDOM_DATA_FILE_SIZE > length) {
			accumulator = new unsigned char[length];
			file.read((char*)accumulator, length);
			accLength = length;
		}
		else {
            accumulator = new unsigned char[RANDOM_DATA_FILE_SIZE];
            file.read((char*)accumulator, RANDOM_DATA_FILE_SIZE);
            accLength = RANDOM_DATA_FILE_SIZE;
		}
		file.close();
	}
	else {
        accumulator = new unsigned char[RANDOM_DATA_FILE_SIZE];
        accLength = RANDOM_DATA_FILE_SIZE;
		// MAX 32B INT = 256x256x256x256
		accumulator[0] = rand()%256;
		accumulator[1] = rand()%256;
		accumulator[2] = rand()%256;
		accumulator[3] = rand()%256;
		bQRGBS = false;
	}
	accPosition = rand()%(accLength-4);

    return STAT_OK;
}

int CRndGen::UpdateAccumulator() {
	accPosition += 4;
	if ((accPosition+4) > accLength)
		InitRandomGenerator(seed,bQRGBSPath);
	else if (!bQRGBS) {
		//REGENERATE THE ACCUMULATOR
		accumulator[accPosition] = rand()%256;
		accumulator[accPosition+1] = rand()%256;
		accumulator[accPosition+2] = rand()%256;
		accumulator[accPosition+3] = rand()%256;
	}
	
    return STAT_OK;
}  

string CRndGen::ToString() {
	if (bQRGBS) return "QRBS SOURCE";
	else return "SYSTEM SOURCE";
}
