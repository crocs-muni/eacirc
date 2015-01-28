#include "EstreamCiphers.h"

const char* EstreamCiphers::estreamToString(int cipher) {
    switch (cipher) {
    case ESTREAM_ABC:               return "ABC";
    case ESTREAM_ACHTERBAHN:        return "Achterbahn";
    case ESTREAM_CRYPTMT:           return "CryptMT";
    case ESTREAM_DECIM:             return "DECIM";
    case ESTREAM_DICING:            return "DICING";
    case ESTREAM_DRAGON:            return "Dragon";
    case ESTREAM_EDON80:            return "Edon80";
    case ESTREAM_FFCSR:             return "F-FCSR";
    case ESTREAM_FUBUKI:            return "Fubuki";
    case ESTREAM_GRAIN:             return "Grain";
    case ESTREAM_HC128:             return "HC - version 128";
    case ESTREAM_HERMES:            return "Hermes";
    case ESTREAM_LEX:               return "LEX";
    case ESTREAM_MAG:               return "MAG";
    case ESTREAM_MICKEY:            return "MICKEY";
    case ESTREAM_MIR1:              return "Mir-1";
    case ESTREAM_POMARANCH:         return "Pomaranch";
    case ESTREAM_PY:                return "Py";
    case ESTREAM_RABBIT:            return "Rabbit";
    case ESTREAM_SALSA20:           return "Salsa20";
    case ESTREAM_SFINKS:            return "Sfinks";
    case ESTREAM_SOSEMANUK:         return "SOSEMANUK";
    case ESTREAM_TEA:               return "TEA";
    case ESTREAM_TRIVIUM:           return "Trivium";
    case ESTREAM_TSC4:              return "TSC-4";
    case ESTREAM_WG:                return "WG";
    case ESTREAM_YAMB:              return "Yamb";
    case ESTREAM_ZKCRYPT:           return "Zk-Crypt";
    case ESTREAM_RANDOM:            return "random data";
    default:                        return "(unknown stream cipher)";
    }
}
