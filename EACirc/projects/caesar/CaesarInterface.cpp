#include "CaesarInterface.h"
// CAESAR algorithms
#include "aead/aes128gcm/Aes128Gcm.h"
#include "EACglobals.h"

CaesarInterface::CaesarInterface(int a, int nr, int kl, int sml, int pml, int co)
    : m_algorithm(a), m_numRounds(nr), m_keyLength(kl), m_secretMessageNumberLength(sml),
      m_publicMessageNumberLength(pml), m_cipertextOverhead(co) { }

CaesarInterface::~CaesarInterface() { }

int CaesarInterface::getKeyLength() {
    return m_keyLength;
}

int CaesarInterface::getSecretMessageNumberLength() {
    return m_secretMessageNumberLength;
}

int CaesarInterface::getPublicMessageNumberLength() {
    return m_publicMessageNumberLength;
}

int CaesarInterface::getCipertextOverhead() {
    return m_cipertextOverhead;
}

CaesarInterface* getCaesarFunction(int algorithm, int numRounds) {
    switch (algorithm) {
     case CAESAR_AES128CGM: { return new Aes128Gcm(numRounds); break; }
     default:
         mainLogger.out(LOGGER_ERROR) << "Unknown CAESAR algorithm (" << algorithm << ")." << endl;
         return NULL;
     }
}
