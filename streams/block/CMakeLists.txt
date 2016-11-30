add_library(block STATIC EXCLUDE_FROM_ALL
    block_stream
    block_factory
    block_cipher
    # === eSTREAM cipher files ===
    ciphers/tea/block-sync.h
    ciphers/tea/tea.cpp
    )
target_link_libraries(block core)
