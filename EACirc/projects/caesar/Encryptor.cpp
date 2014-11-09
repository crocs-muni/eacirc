#include "Encryptor.h"
#include "EACglobals.h"
#include "generators/IRndGen.h"

// TODO create length fields in pCaesarSettings and use them instead of virtual call on m_cipher

Encryptor::Encryptor()
    : m_cipher(CaesarInterface::getCaesarFunction(pCaesarSettings->algorithm, pCaesarSettings->algorithmRoundsCount)),
      m_key(new bits_t[m_cipher->getKeyLength()]),
      m_ad(new bits_t[pCaesarSettings->adLength]),
      m_smn(new bits_t[m_cipher->getSecretMessageNumberLength()]),
      m_pmn(new bits_t[m_cipher->getPublicMessageNumberLength()]),
      m_decryptedMessage(new bits_t[pCaesarSettings->plaintextLength]),
      m_decryptedSmn(new bits_t[m_cipher->getSecretMessageNumberLength()]),
      m_decryptedMessageLength(0),
      m_setup(false) {
    pCaesarSettings->ciphertextLength = pCaesarSettings->plaintextLength + m_cipher->getCipertextOverhead();
}

Encryptor::~Encryptor() {
    if (m_cipher) { delete m_cipher; m_cipher = NULL; }
    if (m_key) { delete m_key; m_key = NULL; }
    if (m_ad) { delete m_ad; m_ad = NULL; }
    if (m_smn) { delete m_smn; m_smn = NULL; }
    if (m_pmn) { delete m_pmn; m_pmn = NULL; }
    if (m_decryptedMessage) { delete m_decryptedMessage; m_decryptedMessage = NULL; }
    if (m_decryptedSmn) { delete m_decryptedSmn; m_decryptedSmn = NULL; }
}

int Encryptor::setup() {
    // TODO respect settings
    memset(m_ad, 0, pCaesarSettings->adLength);
    memset(m_smn, 0, m_cipher->getSecretMessageNumberLength());
    memset(m_pmn, 0, m_cipher->getPublicMessageNumberLength());
    for (int i = 0; i < m_cipher->getKeyLength(); i++) { rndGen->getRandomFromInterval(255, &(m_key[i])); }

    m_setup = true;
    return STAT_OK;
}

int Encryptor::encrypt(const bits_t *m, bits_t *c, length_t *clen) {
    if (!m_setup) {
        mainLogger.out(LOGGER_ERROR) << "Cipher not properly setup!" << endl;
        return STAT_PROJECT_ERROR;
    }

    int encryptionStatus = 0;
    encryptionStatus = m_cipher->encrypt(c, clen, m, pCaesarSettings->plaintextLength,
                                         m_ad, pCaesarSettings->adLength, m_smn, m_pmn, m_key);
    if (encryptionStatus != 0) {
        mainLogger.out(LOGGER_ERROR) << "Encryption failed (status " << encryptionStatus << ")." << endl;
        return STAT_PROJECT_ERROR;
    }
    encryptionStatus = m_cipher->decrypt(m_decryptedMessage, &m_decryptedMessageLength, m_decryptedSmn,
                                         c, *clen, m_ad, pCaesarSettings->adLength, m_pmn, m_key);
    if (encryptionStatus != 0) {
        mainLogger.out(LOGGER_ERROR) << "Decryption failed (status " << encryptionStatus << ")." << endl;
        return STAT_PROJECT_ERROR;
    }
    if (pCaesarSettings->plaintextLength != m_decryptedMessageLength) {
        mainLogger.out(LOGGER_ERROR) << "Decrypted plaintext length mismatch (" << pCaesarSettings->plaintextLength << " versus " << m_decryptedMessageLength << ")." << endl;
        return STAT_PROJECT_ERROR;
    }
    if (memcmp(m, m_decryptedMessage, m_decryptedMessageLength) != 0) {
        mainLogger.out(LOGGER_ERROR) << "Decrypted message mismatch." << endl;
        return STAT_PROJECT_ERROR;
    }
    if (memcmp(m_smn, m_decryptedSmn, m_cipher->getSecretMessageNumberLength()) != 0) {
        mainLogger.out(LOGGER_ERROR) << "Decrypted secret message number mismatch." << endl;
        return STAT_PROJECT_ERROR;
    }

    return STAT_OK;
}
