cmake_minimum_required(VERSION 3.1)

project(manifold_cdkf VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

option(BUILD_EXAMPLES "Build the examples." ON)
option(BUILD_TESTS "Build the tests." OFF)

find_package(Eigen3 REQUIRED CONFIG)

###########
## Build ##
###########

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
if(TARGET Eigen3::Eigen)
  target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)
else() # Older Eigen versions
  target_include_directories(${PROJECT_NAME} SYSTEM INTERFACE
    ${EIGEN3_INCLUDE_DIRS})
endif()
target_compile_options(${PROJECT_NAME} INTERFACE "-std=c++14")

if(BUILD_EXAMPLES)
  add_executable(pose_vel_example src/pose_vel_example.cpp)
  target_link_libraries(pose_vel_example ${PROJECT_NAME})
endif()

#############
## Install ##
#############

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
  INTERFACE_LIBRARIES manifold_cdkf
  DEPS Eigen3
  DEPS_INCLUDE_DIRS EIGEN3
  DEPS_LIBRARIES
  )

#############
## Testing ##
#############

if(BUILD_TESTS)
  add_executable(test_unit_vector_3d tests/test_unit_vector_3d.cpp)
  target_include_directories(test_unit_vector_3d PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  if(TARGET Eigen3::Eigen)
    target_link_libraries(test_unit_vector_3d Eigen3::Eigen)
  else() # Older Eigen versions
    target_include_directories(test_unit_vector_3d SYSTEM PUBLIC
      ${EIGEN3_INCLUDE_DIRS})
  endif()
  add_executable(test_so2 tests/test_so2.cpp)
  target_include_directories(test_so2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  if(TARGET Eigen3::Eigen)
    target_link_libraries(test_so2 Eigen3::Eigen)
  else() # Older Eigen versions
    target_include_directories(test_so2 SYSTEM PUBLIC
      ${EIGEN3_INCLUDE_DIRS})
  endif()
  enable_testing()
  add_test(UnitVector3D test_unit_vector_3d)
  add_test(SO2 test_so2)
endif()
