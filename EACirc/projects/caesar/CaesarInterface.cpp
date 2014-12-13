#include "CaesarInterface.h"
#include "EACglobals.h"

// CAESAR algorithms
#include "aead/AES-GCM/AesGcm.h"
#include "aead/ACORN/Acorn.h"
#include "aead/OCB/Ocb.h"

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
     case CAESAR_AESGCM: { return new AesGcm(numRounds); break; }
     case CAESAR_ACORN: { return new Acorn(numRounds); break; }
    case CAESAR_OCB: { return new Ocb(numRounds, mode); break; }
     default:
         mainLogger.out(LOGGER_ERROR) << "Unknown CAESAR algorithm (" << algorithm << ")." << endl;
         return NULL;
     }
}
