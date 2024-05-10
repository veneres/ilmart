cmake_minimum_required(VERSION 3.18)
project(fast_distilled)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

set(CMAKE_CXX_STANDARD 20)

add_executable(fast_distilled main.cpp simple/ilmart_scorer.cpp utilities.cpp)
