# AES-GCM

**Designers:** David A. McGrew, John Viega

**Implementation:** reference
**Implemetors:** Daniel J. Bernstein
**Version:** v1
**Source:** https://github.com/floodyberry/supercop/tree/master/crypto_aead/aes128gcmv1
**Download date:** 2014-11-01

## Available modes

* key 128 bits, tag 128 bits

## Changes to the code

* File renamed to cpp, name prefixed.
* Includes sorted out.
* Common includes changed to '../common/api.h'.
* Namespace added.
* Variable numRounds added.
* Functions 'crypto_core_aes128encrypt' and 'crypto_verify_16' called from namespace CaesarCommon.
