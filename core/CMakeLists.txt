add_library(core STATIC
    bitset
    cmd
    debug
    json
    lazy
    logger
    maybe
    random
    range
    stream
    traits
    utils
    variant
    )

target_include_directories(core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

set_target_properties(core PROPERTIES
    LINKER_LANGUAGE CXX
    )

target_link_libraries(core
    json
    pcg
    )
