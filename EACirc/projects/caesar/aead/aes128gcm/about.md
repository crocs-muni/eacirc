# AES-128-GCM v1

**Designers:** David A. McGrew, John Viega

**Implementation:** reference
**Implemetors:** Daniel J. Bernstein
**Version:** v1
**Source:** https://github.com/floodyberry/supercop/tree/master/crypto_aead/aes128gcmv1**Download date:** 2014-11-01

## Changes to the code

* crypto_aead.h include changed to encrypt.h
* Common includes path changed.
* Namespace added.
* Varible numRounds added.
* Functions 'crypto_core_aes128encrypt' and 'crypto_verify_16' called from namespace CaesarCommon.
