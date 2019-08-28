cmake_minimum_required(VERSION 3.2.0)

project(lightrnn)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(Eigen3_DIR "/search/odin/imer/packages/eigen-3.3.4/cmake")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

file(GLOB_RECURSE LIGHTRNN_CPP "${PROJECT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE MAIN_CPP "${PROJECT_SOURCE_DIR}/main.cpp")
file(GLOB_RECURSE REMOVE_CMAKE "${PROJECT_SOURCE_DIR}/build/*")
list(REMOVE_ITEM LIGHTRNN_CPP ${MAIN_CPP})
list(REMOVE_ITEM LIGHTRNN_CPP ${REMOVE_CMAKE})
file(GLOB_RECURSE LIGHTRNN_H   "${PROJECT_SOURCE_DIR}/*.h")

option (build_Library "Build shared object file" OFF)
option (build_lightrnn_test "Build test executable file" ON)


if(build_Library)
	add_library(${PROJECT_NAME})
endif()

if(build_lightrnn_test)
	add_executable("${PROJECT_NAME}_test" ${LIGHTRNN_CPP} ${LIGHTRNN_H} ${MAIN_CPP})
	target_link_libraries ("${PROJECT_NAME}_test" Eigen3::Eigen)
endif()

