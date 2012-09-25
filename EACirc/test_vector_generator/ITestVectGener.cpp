#include "SSGlobals.h"
#include "test_vector_generator/ITestVectGener.h"
#include "test_vector_generator/EstreamVectGener.h"
#include "EACirc.h"

ITestVectGener::ITestVectGener() {
}

ITestVectGener* ITestVectGener::getGenerClass(void) {
	switch (pGACirc->testVectorGenerMethod) {
		case ESTREAM_CONST:
			return new EstreamTestVectGener();
			break;
		default:
            assert(FALSE);
			break;
	}
	return NULL;
}
