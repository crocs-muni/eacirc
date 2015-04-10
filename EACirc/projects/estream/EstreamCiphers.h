#ifndef ESTREAMCIPHERS_H
#define ESTREAMCIPHERS_H

// eStream cipher constants
#define ESTREAM_ABC             1
#define ESTREAM_ACHTERBAHN      2
#define ESTREAM_CRYPTMT         3
#define ESTREAM_DECIM           4
#define ESTREAM_DICING          5
#define ESTREAM_DRAGON          6
#define ESTREAM_EDON80          7
#define ESTREAM_FFCSR           8
#define ESTREAM_FUBUKI          9
#define ESTREAM_GRAIN           10
#define ESTREAM_HC128           11
#define ESTREAM_HERMES          12
#define ESTREAM_LEX             13
#define ESTREAM_MAG             14
#define ESTREAM_MICKEY          15
#define ESTREAM_MIR1            16
#define ESTREAM_POMARANCH       17
#define ESTREAM_PY              18
#define ESTREAM_RABBIT          19
#define ESTREAM_SALSA20         20
#define ESTREAM_SFINKS          21
#define ESTREAM_SOSEMANUK       22
#define ESTREAM_TEA             23
#define ESTREAM_TRIVIUM         24
#define ESTREAM_TSC4            25
#define ESTREAM_WG              26
#define ESTREAM_YAMB            27
#define ESTREAM_ZKCRYPT         28
#define ESTREAM_RANDOM          99

namespace EstreamCiphers {

    /** converts eStream cipher constant to human-readable string
      * @param cipher       cipher constant
      * @return human readable cipher description
      */
    const char* estreamToString(int cipher);

} // end namespace EstreamCiphers

#endif // ESTREAMCIPHERS_H
