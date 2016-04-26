add_library(core STATIC
    base bitset compiletime dataset exceptions logger project teestream utils)

target_include_directories(core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
