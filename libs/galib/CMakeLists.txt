# === Galib cmake file ===
cmake_minimum_required(VERSION 3.0.2)


include(${PROJECT_SOURCE_DIR}/cmake/remove_cxx_flags.cmake)
remove_cxx_flags(SUPRESS_LIBS_WARNINGS)


set(SOURCES
    garandom.cpp
    gaerror.cpp
    GAParameter.cpp
    GAStatistics.cpp
    GABaseGA.cpp
    GASStateGA.cpp
    GASimpleGA.cpp
    GAIncGA.cpp
    GADemeGA.cpp
    GADCrowdingGA.cpp
    GASelector.cpp
    GAScaling.cpp
    GAPopulation.cpp
    GAGenome.cpp
    GABinStr.cpp
    gabincvt.cpp
    GAAllele.cpp
    GAStringGenome.cpp
    GARealGenome.cpp
    GA1DBinStrGenome.cpp
    GA2DBinStrGenome.cpp
    GA3DBinStrGenome.cpp
    GABin2DecGenome.cpp
    GA1DArrayGenome.cpp
    GA2DArrayGenome.cpp
    GA3DArrayGenome.cpp
    GATreeBASE.cpp
    GATree.cpp
    GATreeGenome.cpp
    GAListBASE.cpp
    GAList.cpp
    GAListGenome.cpp
    GAAllele.cpp
    GA1DArrayGenome.cpp
    GA2DArrayGenome.cpp
    GA3DArrayGenome.cpp
    GATree.cpp
    GATreeGenome.cpp
    GAList.cpp
    GAListGenome.cpp
    GAStringGenome.cpp
    GARealGenome.cpp
)


set(HEADERS
    GA1DArrayGenome.h
    GA1DBinStrGenome.h
    GA2DArrayGenome.h
    GA2DBinStrGenome.h
    GA3DArrayGenome.h
    GA3DBinStrGenome.h
    GAAllele.h
    GAArray.h
    GABaseGA.h
    GABin2DecGenome.h
    gabincvt.h
    GABinStr.h
    gaconfig.h
    GADCrowdingGA.h
    GADemeGA.h
    gaerror.h
    GAEvalData.h
    GAGenome.h
    ga.h
    gaid.h
    GAIncGA.h
    GAListBASE.h
    GAListGenome.h
    GAList.h
    GAMask.h
    GANode.h
    GAParameter.h
    GAPopulation.h
    garandom.h
    GARealGenome.h
    GAScaling.h
    GASelector.h
    GASimpleGA.h
    GASStateGA.h
    GAStatistics.h
    GAStringGenome.h
    GATreeBASE.h
    GATreeGenome.h
    GATree.h
    gatypes.h
    gaversion.h
    std_stream.h
)


add_library(galib STATIC ${SOURCES} ${HEADERS})

target_include_directories(galib
    PUBLIC INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(galib
    PROPERTIES FOLDER libs
)
