cmake_minimum_required(VERSION 3.0)

project(manifold_cdkf VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)
target_compile_options(${PROJECT_NAME} INTERFACE "-std=c++17")

add_executable(pose_vel_example src/pose_vel_example.cpp)
target_link_libraries(pose_vel_example Eigen3::Eigen ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} EXPORT "${PROJECT_NAME}Targets"
  LIBRARY DESTINATION "lib"
  ARCHIVE DESTINATION "lib"
  RUNTIME DESTINATION "bin"
  INCLUDES DESTINATION "include"
  )

install(DIRECTORY "include/${PROJECT_NAME}"
  DESTINATION "include"
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
  )

include(cmake/GenerateAndInstallConfig.cmake)
generate_and_install_config_file(
  INCLUDE_DIRS "include"
  LIBRARIES
  DEPS Eigen3
  DEPS_INCLUDE_DIRS EIGEN3
  DEPS_LIBRARIES
  )
