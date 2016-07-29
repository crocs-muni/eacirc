add_library(core STATIC
    #ea-iterators
    #ea-json
    #ea-logger
    #ea-settings
    #ea-utils
    bitset
    debug
    lazy
    maybe
    random
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
