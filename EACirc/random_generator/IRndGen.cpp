#include "SSGlobals.h"
#include "IRndGen.h"
#include "RndGen.h"
#include "BiasRndGen.h"
#include "EACirc.h"

IRndGen::IRndGen() {
	this->type = CRNDGEN;
}

IRndGen::IRndGen(int type) {
	this->type = type;
}

IRndGen* IRndGen::getRndGenClass(void) {
	switch (type) {
		case CRNDGEN:
			return new CRndGen();
			break;
		case BIASGEN: 
			return new BiasRndGen();
			break;
		default:
            assert(FALSE);
			break;
	}
	return NULL;
}
