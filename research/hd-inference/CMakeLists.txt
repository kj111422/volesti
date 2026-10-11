project(hd_inference)
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DDISABLE_NLP_ORACLES)
# Usage: cmake -DVOLESTI_DIR=<path to volesti checkout> ..
# Defaults to the enclosing volesti checkout.
if(NOT DEFINED VOLESTI_DIR)
  set(VOLESTI_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()
get_filename_component(VOLESTI_DIR "${VOLESTI_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")
include("${VOLESTI_DIR}/external/cmake-files/Eigen.cmake")
GetEigen()
include("${VOLESTI_DIR}/external/cmake-files/Boost.cmake")
GetBoost()
include("${VOLESTI_DIR}/external/cmake-files/LPSolve.cmake")
GetLPSolve()
include_directories(
    BEFORE
    ${VOLESTI_DIR}/external/Spectra/include
    ${VOLESTI_DIR}/include
    ${VOLESTI_DIR}/external
    ${VOLESTI_DIR}/external/minimum_ellipsoid
)
add_executable(reference_tests reference_tests.cpp)
target_link_libraries(reference_tests PUBLIC lp_solve)
add_executable(phase2a phase2a.cpp)
add_executable(phase2b phase2b.cpp)
target_link_libraries(phase2b PUBLIC lp_solve)
