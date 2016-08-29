add_library(core STATIC
    cmd
    debug
    json
    logger
    random
    stream
    traits
    variant
    view
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
