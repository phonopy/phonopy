cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0007 NEW)

project(phonopy C)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
set(CMAKE_C_FLAGS_DEBUG "-g -DTHMWARNING")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SYSTEM_PREFIX_PATH: ${CMAKE_SYSTEM_PREFIX_PATH}")

include(GNUInstallDirs)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/phonopy/version.py version_file)
string(REGEX MATCH "__version__.*([0-9]+)[.]([0-9]+)[.]([0-9]+)" phono3py_version ${version_file})
set(phonopy_major_version ${CMAKE_MATCH_1})
set(phonopy_minor_version ${CMAKE_MATCH_2})
set(phonopy_micro_version ${CMAKE_MATCH_3})
set(serial "${phonopy_major_version}.${phonopy_minor_version}.${phonopy_micro_version}")
set(soserial "1")

find_package(OpenMP)
if (OpenMP_FOUND)
  message(STATUS "OpenMP libs: ${OpenMP_C_LIBRARIES}")
  message(STATUS "OpenMP flags: ${OpenMP_C_FLAGS}")
endif()

if (PHONOPY)
  set(SOURCES_PHONOPY
    ${PROJECT_SOURCE_DIR}/c/phonopy.c
    ${PROJECT_SOURCE_DIR}/c/dynmat.c
    ${PROJECT_SOURCE_DIR}/c/derivative_dynmat.c
    ${PROJECT_SOURCE_DIR}/c/rgrid.c
    ${PROJECT_SOURCE_DIR}/c/tetrahedron_method.c)

  if (BUILD_SHARED_LIBRARIES)
  # Shared library
    add_library(phpy SHARED ${SOURCES_PHONOPY})
    if (OpenMP_FOUND)
      target_link_libraries(phpy m ${OpenMP_C_LIBRARIES})
      target_compile_options(phpy PRIVATE ${OpenMP_C_FLAGS})
    else()
      target_link_libraries(phpy m)
    endif()
    target_include_directories(phpy PRIVATE ${MY_INCLUDES})
    target_compile_definitions(phpy PRIVATE THM_EPSILON=1e-10)
    set_property(TARGET phpy PROPERTY VERSION ${serial})
    set_property(TARGET phpy PROPERTY SOVERSION ${soserial})
    install(TARGETS phpy LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  # Static link library
  add_library(phpy_static STATIC ${SOURCES_PHONOPY})
  if (OpenMP_FOUND)
    target_link_libraries(phpy_static m ${OpenMP_C_LIBRARIES})
    target_compile_options(phpy_static PRIVATE ${OpenMP_C_FLAGS})
  else()
    target_link_libraries(phpy_static m)
  endif()
  target_include_directories(phpy_static PRIVATE ${MY_INCLUDES})
  target_compile_definitions(phpy_static PRIVATE THM_EPSILON=1e-10)
  set_property(TARGET phpy_static PROPERTY VERSION ${serial})
  set_property(TARGET phpy_static PROPERTY SOVERSION ${soserial})
  set_property(TARGET phpy_static PROPERTY OUTPUT_NAME phpy)
  install(TARGETS phpy_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Header file
  install(FILES ${PROJECT_SOURCE_DIR}/c/phonopy.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
