# === tinyXML cmake file ===
add_library(tinyXML STATIC
    tinystr tinyxml tinyxmlerror tinyxmlparser)

target_include_directories(tinyXML
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(tinyXML
    PROPERTIES FOLDER libs
)
