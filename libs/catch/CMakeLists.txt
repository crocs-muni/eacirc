# === Catch framework cmake file ===
cmake_minimum_required(VERSION 3.0.2)


include(${PROJECT_SOURCE_DIR}/cmake/remove_cxx_flags.cmake)
remove_cxx_flags(SUPRESS_LIBS_WARNINGS)


add_library(catch INTERFACE)

target_include_directories(catch
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
