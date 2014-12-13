# OCB

**Designers:** Ted Krovetz, Phillip Rogaway

**Implementation:** reference
**Implemetors:** Ted Krovetz
**Version:** v1
**Source:** https://github.com/floodyberry/supercop/tree/master/crypto_aead/aeadaes128ocbtaglen128v1
**Download date:** 2014-12-13

## Available modes

1) key 128 bits, tag 64 bits
2)key 128 bits, tag 96 bits
3) key 128 bits, tag 128 bits
4) key 192 bits, tag 64 bits
5) key 192 bits, tag 96 bits
6) key 192 bits, tag 128 bits
7) key 256 bits, tag 64 bits
8) key 256 bits, tag 96 bits
9) key 256 bits, tag 128 bits

## Changes to the code

* File renamed to cpp, name prefixed.
* Includes sorted out.
* Namespace added.
* Variable numRounds, mode added.
* File 'api.h' changed to account for different modes.
