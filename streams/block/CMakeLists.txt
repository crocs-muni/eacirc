add_library(block STATIC EXCLUDE_FROM_ALL
    block_stream
    block_factory
    block_interface
    # === eSTREAM cipher files ===
    ciphers/block-config.h
    ciphers/block-machine.h
    ciphers/block-portable.h
    ciphers/tea/block-sync.h
    )
target_link_libraries(block core)
