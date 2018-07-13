function(generate_and_install_config_file)
  cmake_parse_arguments(config "" "" "INCLUDE_DIRS;LIBRARIES;DEPS;DEPS_INCLUDE_DIRS;DEPS_LIBRARIES" ${ARGN})

  # Configuration (https://github.com/forexample/package-example)
  set(config_install_dir "share/${PROJECT_NAME}/cmake")
  set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
  set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file("${version_config}"
    COMPATIBILITY SameMajorVersion
    )

  foreach(dep ${config_DEPS})
    set(PROJECT_DEPS "${PROJECT_DEPS}\nfind_dependency(${dep})")
  endforeach()

  foreach(dir ${config_INCLUDE_DIRS})
    if(IS_ABSOLUTE ${dir} AND EXISTS ${dir})
      set(CUR_DIR ${dir})
    else()
      set(CUR_DIR "\${PACKAGE_PREFIX_DIR}/${dir}")
    endif()
    list(APPEND PROJECT_INCLUDE_DIRS ${CUR_DIR})
    set(CUR_DIR)
  endforeach()
  foreach(dep ${config_DEPS_INCLUDE_DIRS})
    list(APPEND PROJECT_INCLUDE_DIRS "\${${dep}_INCLUDE_DIRS}")
  endforeach()
  list(LENGTH PROJECT_INCLUDE_DIRS PROJECT_INCLUDE_DIRS_LENGTH)
  if(${PROJECT_INCLUDE_DIRS_LENGTH})
    list(REMOVE_DUPLICATES PROJECT_INCLUDE_DIRS)
  endif()

  foreach(lib ${config_LIBRARIES})
    if(IS_ABSOLUTE ${lib} AND EXISTS ${lib})
      set(CUR_LIB ${lib})
    else()
      set(CUR_LIB "\${PACKAGE_PREFIX_DIR}/lib/lib${lib}.so")
    endif()
    list(APPEND PROJECT_LIBRARIES ${CUR_LIB})
    set(CUR_LIB)
  endforeach()
  foreach(dep ${config_DEPS_LIBRARIES})
    list(APPEND PROJECT_LIBRARIES "\${${dep}_LIBRARIES}")
  endforeach()
  list(LENGTH PROJECT_LIBRARIES PROJECT_LIBRARIES_LENGTH)
  if(${PROJECT_LIBRARIES_LENGTH})
    list(REMOVE_DUPLICATES PROJECT_LIBRARIES)
  endif()
  #configure_file("cmake/Config.cmake.in" "${project_config}" @ONLY)

  configure_package_config_file(
    "cmake/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
    NO_SET_AND_CHECK_MACRO
    )

  install(FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
    )

  install(EXPORT "${PROJECT_NAME}Targets"
    NAMESPACE "${PROJECT_NAME}::"
    DESTINATION "${config_install_dir}"
    )
endfunction()
