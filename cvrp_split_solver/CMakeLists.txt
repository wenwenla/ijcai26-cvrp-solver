cmake_minimum_required(VERSION 3.12)
project(splitting_solver)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python3_LIBRARIES}")

pybind11_add_module(gmsvrprf general_solver_rf.cpp)
