#include "SSGlobals.h"
#include "ITestVectGener.h"
#include "EstreamVectGener.h"
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