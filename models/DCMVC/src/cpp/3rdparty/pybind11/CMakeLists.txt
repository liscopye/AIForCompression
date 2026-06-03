# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(NOT PYBIND11_INCLUDE)
    find_package(pybind11 CONFIG REQUIRED)
    set(PYBIND11_INCLUDE ${pybind11_INCLUDE_DIRS} CACHE INTERNAL "")
endif()
