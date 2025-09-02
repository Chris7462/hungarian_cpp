cmake_minimum_required(VERSION 3.16)

project(hungarian
  VERSION 1.0.0
  DESCRIPTION "Modern C++ implementation of Hungarian Algorithm using Eigen"
  LANGUAGES CXX)

# Option to enable/disable ROS 2 support
option(ENABLE_ROS2 "Build with ROS 2 support" ON)

# Auto-detect ROS 2 environment if not explicitly set
if(ENABLE_ROS2 AND NOT DEFINED ENV{ROS_DISTRO})
  message(STATUS "ROS_DISTRO not found, disabling ROS 2 support")
  set(ENABLE_ROS2 OFF)
endif()

# Try to find ament_cmake
if(ENABLE_ROS2)
  find_package(ament_cmake QUIET)
  if(NOT ament_cmake_FOUND)
    message(STATUS "ament_cmake not found, building as pure CMake")
    set(ENABLE_ROS2 OFF)
  endif()
endif()

message(STATUS "Building with ROS 2 support: ${ENABLE_ROS2}")

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Library configuration
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

add_library(${PROJECT_NAME}
  src/hungarian.cpp)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

# =============================================================================
# ROS 2 SPECIFIC CONFIGURATION
# =============================================================================
if(ENABLE_ROS2)
  message(STATUS "Configuring for ROS 2 build")

  # Install targets
  install(TARGETS ${PROJECT_NAME}
    EXPORT export_${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

  # Install headers
  install(DIRECTORY include/${PROJECT_NAME}
    DESTINATION include)

  # Export targets for ROS 2
  ament_export_targets(${EXPORT_NAME} HAS_LIBRARY_TARGET)
  ament_export_dependencies(Eigen3)

  # ROS 2 Testing
  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_copyright_FOUND TRUE)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()

    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_${PROJECT_NAME} test/test_hungarian.cpp)
    target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME})
  endif()

  ament_package()

# =============================================================================
# PURE CMAKE CONFIGURATION
# =============================================================================
else()
  message(STATUS "Configuring for pure CMake build")

  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # Install targets
  install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # Install headers
  install(DIRECTORY include/${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # Export targets for pure CMake
  install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  # Create config files
  configure_package_config_file(
    cmake/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  # Add uninstall target
  if(NOT TARGET uninstall)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
      IMMEDIATE @ONLY)

    add_custom_target(uninstall
      COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
  endif()

  # Pure CMake Testing
  if(BUILD_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(test_${PROJECT_NAME} test/test_hungarian.cpp)
    target_link_libraries(test_${PROJECT_NAME}
      ${PROJECT_NAME}
      GTest::gtest_main)

    gtest_discover_tests(test_${PROJECT_NAME})
  endif()
endif()
