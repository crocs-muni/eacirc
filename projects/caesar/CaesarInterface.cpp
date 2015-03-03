#include "CaesarInterface.h"
#include "EACglobals.h"

// CAESAR algorithms
#include "aead/aes128gcm/Aes128Gcm.h"
#include "aead/acorn128/Acorn128.h"

CaesarInterface::CaesarInterface(int a, int nr, int kl, int smnl, int pmnl, int co)
    : m_algorithm(a), m_numRounds(nr) {
    pCaesarSettings->keyLength = kl;
    pCaesarSettings->smnLength = smnl;
    pCaesarSettings->pmnLength = pmnl;
    pCaesarSettings->cipertextOverhead = co;
}

CaesarInterface::~CaesarInterface() { }

CaesarInterface* CaesarInterface::getCaesarFunction(int algorithm, int numRounds) {
    switch (algorithm) {
     case CAESAR_AES128CGM: { return new Aes128Gcm(numRounds); break; }
     case CAESAR_ACORN128: { return new Acorn128(numRounds); break; }
     default:
         mainLogger.out(LOGGER_ERROR) << "Unknown CAESAR algorithm (" << algorithm << ")." << endl;
         return NULL;
     }
}
