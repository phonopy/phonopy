cmake_minimum_required(VERSION 3.25...3.31)

#[=============================================================================[
#                           Basic project definition                           #
]=============================================================================]

list(APPEND CMAKE_MESSAGE_CONTEXT phonopy)
project(phonopy
        VERSION ${SKBUILD_PROJECT_VERSION}
        DESCRIPTION "Phonon code mainly written in python"
        HOMEPAGE_URL https://github.com/phonopy/phonopy
        LANGUAGES C CXX
)

# Specify C/C++ standard used in the project.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#[=============================================================================[
#                                   Options                                   #
]=============================================================================]

include(CMakeDependentOption)
include(FeatureSummary)

option(PHONOPY_SHARED_LIBS "Phonopy: Build as a shared library" ${PROJECT_IS_TOP_LEVEL})
option(PHONOPY_USE_OMP "Phonopy: Build with OpenMP support" ON)

#[=============================================================================[
#                            Project configuration                            #
]=============================================================================]

include(GNUInstallDirs)

set(BUILD_SHARED_LIBS ${PHONOPY_SHARED_LIBS})

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#[=============================================================================[
#                              External packages                              #
]=============================================================================]

if (PHONOPY_USE_OMP)
    find_package(OpenMP REQUIRED COMPONENTS C CXX)
endif ()
find_package(Python 3.9 REQUIRED
        COMPONENTS Development.Module
)
find_package(nanobind REQUIRED CONFIG)

#[=============================================================================[
#                               Main definition                               #
]=============================================================================]

add_library(phonopy_lib)
set_target_properties(phonopy_lib PROPERTIES
        VERSION ${PROJECT_VERSION}
        OUTPUT_NAME phonopy
)
nanobind_add_module(_phonopy)
add_subdirectory(c)
