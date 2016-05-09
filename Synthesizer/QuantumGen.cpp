#include "QuantumGen.h"
#include <iostream>


void QuantumRndGen::read(Dataset& data) {
	u8* dataPtr = data.data();
	int byteSize = data.num_of_tvs()*data.tv_size();

	for (int i = 0; i < byteSize; i ++)
	{
		dataPtr[i] = operator()();
	}
}

QuantumRndGen::QuantumRndGen(unsigned long seed, std::string fileName) {
	engine.seed(seed);
	if(!loadQRNGDataFile(fileName)) std::cout << "file not loaded";
	dis = std::uniform_int_distribution<>(0, qrngData.size()-1);

}

u8 QuantumRndGen::operator()() {
	return qrngData[dis(engine)];
}

bool QuantumRndGen::loadQRNGDataFile(std::string fileName) {
	std::ifstream file;
	int length; 
	file.open(fileName.c_str(), std::fstream::in | std::fstream::binary);
	if (file.is_open()) {
		// DETERMINE DATA LENGTH
		file.seekg(0, std::ios::end);
		length = file.tellg();
		file.seekg(0, std::ios::beg);
		qrngData.resize(length);
		file.read((char*)&qrngData[0], length);
		file.close();
		return true;
	}
	else {
		return false;
	}
	


}


