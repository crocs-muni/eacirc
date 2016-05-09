#ifndef QUANTUM_RNDGEN_H
#define QUANTUM_RNDGEN_H

#include<vector>
#include<fstream>
#include"../core/base.h"
#include"../core/project.h"
#include "RandGen.h"
#include<random>
#include<string>

// polynomial is evaluated to zero - at least one random variable in each term is set to zero

class QuantumRndGen : public  Stream, RandGen<u8> {
public:
	QuantumRndGen(unsigned long seed, std::string fileName);
	void read(Dataset& data);
	u8 operator()();
	bool loadQRNGDataFile(std::string fileName);
private:
	std::mt19937 engine;
	std::uniform_int_distribution<int> dis;
	std::vector<u8> qrngData;
};
#endif //QUANTUM_RNDGEN_H
