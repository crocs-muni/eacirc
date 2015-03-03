# === caesar ===
cmake_minimum_required(VERSION 3.0.2)

if(NOT BUILD_CAESAR)
    return()
endif()


set(SOURCES
    CaesarProject.cpp
    CaesarInterface.cpp
    Encryptor.cpp
# === CAESAR algorithms ===
    aead/common/crypto_verify_16.cpp
    aead/common/crypto_core_aes128encrypt.cpp
    aead/aes128gcm/Aes128Gcm_encrypt.cpp
    aead/aes128gcm/Aes128Gcm.cpp
    aead/acorn128/Acorn128.cpp
    aead/acorn128/Acorn128_encrypt.cpp
)

set(HEADERS
    CaesarConstants.h
    CaesarProject.h
    CaesarInterface.h
    Encryptor.h
# === CAESAR algorithms ===
    aead/common/api.h
    aead/aes128gcm/Aes128Gcm.h
    aead/aes128gcm/api.h
    aead/aes128gcm/Aes128Gcm_encrypt.h
    aead/acorn128/Acorn128.h
    aead/acorn128/api.h
    aead/acorn128/Acorn128_encrypt.h
)

set(OTHER_FILES
    aead/acorn128/about.md
    aead/aes128gcm/about.md
)


add_library(caesar STATIC ${SOURCES} ${HEADERS})
target_link_libraries(caesar tinyXML)
target_include_directories(caesar PRIVATE ${PROJECT_SOURCE_DIR}/EACirc)