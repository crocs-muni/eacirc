#include "CaesarInterface.h"
#include "EACglobals.h"

// CAESAR algorithms
#include "aead/aes128gcmv1/Aes128gcmv1.h"
#include "aead/acorn128/Acorn128.h"

CaesarInterface::CaesarInterface(int a, int nr, int kl, int smnl, int pmnl, int co)
    : m_algorithm(a), m_numRounds(nr) {
    pCaesarSettings->keyLength = kl;
    pCaesarSettings->smnLength = smnl;
    pCaesarSettings->pmnLength = pmnl;
    pCaesarSettings->cipertextOverhead = co;
}

CaesarInterface::~CaesarInterface() { }

CaesarInterface* CaesarInterface::getCaesarFunction(int algorithm, int numRounds, int mode) {
    switch (algorithm) {
     case CAESAR_AESGCM: { return new Aes128gcmv1(numRounds); break; }
     case CAESAR_ACORN: { return new Acorn128(numRounds); break; }
     //case CAESAR_OCB: { return new Ocb(numRounds, mode); break; }
     default:
         mainLogger.out(LOGGER_ERROR) << "Unknown CAESAR algorithm (" << algorithm << ")." << endl;
         return NULL;
     }
}
