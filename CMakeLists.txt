cmake_minimum_required(VERSION 3.15)
project(gpu_scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(gpu_scheduler
    cpp/src/cluster.cpp
    cpp/src/scheduler.cpp
    cpp/src/simulator.cpp
    cpp/src/bindings.cpp
)

target_include_directories(gpu_scheduler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

install(TARGETS gpu_scheduler LIBRARY DESTINATION .)
