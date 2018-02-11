cmake_minimum_required(VERSION 3.5.1)
project(manifold_cdkf)

find_package(catkin_simple REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_simple()

# From https://lefticus.gitbooks.io/cpp-best-practices/content/02-Use_the_Tools_Available.html#gcc--clang
set(COMPILER_WARNING_FLAGS -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic
  -Wcast-align -Woverloaded-virtual -Wconversion -Wmisleading-indentation
  -Wduplicated-cond -Wlogical-op -Wnull-dereference -Wuseless-cast
  -Wdouble-promotion -Wformat=2 -Wsuggest-final-types
  -Wno-unused-parameter)#-Wsign-conversion -Weffc++)

add_compile_options(-std=c++17 -Wall ${COMPILER_WARNING_FLAGS})

add_executable(pose_vel_example src/pose_vel_example.cpp)
target_link_libraries(pose_vel_example Eigen3::Eigen ${catkin_LIBRARIES})

cs_install()

cs_export(
  DEPENDS EIGEN3
  CFG_EXTRAS ${PROJECT_NAME}-extras.cmake)
