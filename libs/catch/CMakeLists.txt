# === Catch framework cmake file ===
cmake_minimum_required(VERSION 3.0.2)


add_library(catch INTERFACE)

target_include_directories(catch
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)