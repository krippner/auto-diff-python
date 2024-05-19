cmake_minimum_required(VERSION 3.15)
project(AutoDiffPython VERSION 0.1.0 LANGUAGES CXX)

set(PY_VERSION_SUFFIX "")
set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})

# Make sure that the Python and CMake versions match
if (DEFINED PY_BUILD_CMAKE_PACKAGE_VERSION)
    if (NOT "${PY_BUILD_CMAKE_PACKAGE_VERSION}" MATCHES "^${PY_FULL_VERSION}$")
        message(FATAL_ERROR "Version number does not match "
                            "(${PY_BUILD_CMAKE_PACKAGE_VERSION} - ${PY_FULL_VERSION}).")
    endif()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG)
find_package(Eigen3 REQUIRED CONFIG)

include(FetchContent)
FetchContent_Declare(
  autodiff
  GIT_REPOSITORY https://github.com/krippner/auto-diff.git
  GIT_TAG v0.4.0
)
FetchContent_MakeAvailable(autodiff)

option(WITH_PY_STUBS
    "Generate Python stub files (.pyi) for the Python modules." On
)
if (WITH_PY_STUBS AND NOT CMAKE_CROSSCOMPILING)
    include(CMake/stubgen.cmake)
endif()

add_subdirectory(src)
