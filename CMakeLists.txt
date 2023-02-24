cmake_minimum_required(VERSION 3.20)
project(k_means)

set(CMAKE_CXX_STANDARD 17)
# openMP 配置
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_executable(k_means KMeans.cpp KMeans.hpp Point.hpp Point.cpp IOController.cpp IOController.hpp main.cpp )
