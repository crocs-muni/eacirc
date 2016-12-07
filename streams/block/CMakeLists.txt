add_library(block STATIC EXCLUDE_FROM_ALL
    block_stream
    block_cipher
    block_factory.h
    ciphers/common_fun.h
    # === eSTREAM cipher files ===
    ciphers/tea/tea
    ciphers/aes/aes
    ciphers/rc4/rc4
    )
target_link_libraries(block core)
