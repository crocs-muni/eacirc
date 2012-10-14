#ifndef TEST_VECT_GENER_INTERFACE_H
#define TEST_VECT_GENER_INTERFACE_H

#include "EACglobals.h"

class ITestVectGener {
	public:
		ITestVectGener();
		ITestVectGener* getGenerClass(void);
		virtual void generateTestVectors(void) {}
};

#endif