add_library(core STATIC
    base bitset compiletime exceptions factory iterators logger project range teestream testvectors utils)

target_include_directories(core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
