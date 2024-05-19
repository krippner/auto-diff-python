# Add autodiff._scalar module
pybind11_add_module(ScalarLib common.cpp scalar.cpp)
target_compile_definitions(ScalarLib PRIVATE
    MODULE_NAME=$<TARGET_FILE_BASE_NAME:ScalarLib>
    VERSION_INFO="${PY_FULL_VERSION}"
)
target_include_directories(ScalarLib PRIVATE include)
target_link_libraries(ScalarLib PRIVATE AutoDiff::AutoDiff)
set_target_properties(ScalarLib PROPERTIES OUTPUT_NAME "_scalar")

# Add autodiff._array module
pybind11_add_module(ArrayLib common.cpp array.cpp)
target_compile_definitions(ArrayLib PRIVATE
    MODULE_NAME=$<TARGET_FILE_BASE_NAME:ArrayLib>
    VERSION_INFO="${PY_FULL_VERSION}"
)
target_include_directories(ArrayLib PRIVATE include)
target_link_libraries(ArrayLib PRIVATE AutoDiff::AutoDiff Eigen3::Eigen)
set_target_properties(ArrayLib PROPERTIES OUTPUT_NAME "_array")

# Install the modules
install(TARGETS ScalarLib ArrayLib
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME}
)

# Generate stubs for the Python module (autocomplete and type hints)
if (WITH_PY_STUBS AND NOT CMAKE_CROSSCOMPILING)
    pybind11_stubgen(ScalarLib)
    pybind11_stubgen_install(ScalarLib ${PY_BUILD_CMAKE_MODULE_NAME})

    pybind11_stubgen(ArrayLib)
    pybind11_stubgen_install(ArrayLib ${PY_BUILD_CMAKE_MODULE_NAME})
endif()
