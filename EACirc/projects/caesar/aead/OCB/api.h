namespace Ocb_raw {
    extern const unsigned long long keys[];
    extern const unsigned long long abytes[];
} // namespace Ocb_raw

#define CRYPTO_KEYBYTES Ocb_raw::keys[mode-1]
#define CRYPTO_NSECBYTES 0
#define CRYPTO_NPUBBYTES 12
#define CRYPTO_ABYTES Ocb_raw::abytes[mode-1]
