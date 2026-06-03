cmake_minimum_required(VERSION 3.6.3)

project(ryg_rans-download NONE)

include(ExternalProject)
if(EXISTS "/data/run01/scxj523/zsh/project/AIForCompression/models/DCMVC/src/build/3rdparty/ryg_rans/ryg_rans-src/rans64.h")
    ExternalProject_Add(ryg_rans
        GIT_REPOSITORY    https://github.com/rygorous/ryg_rans.git
        GIT_TAG           c9d162d996fd600315af9ae8eb89d832576cb32d
        GIT_SHALLOW       1
        SOURCE_DIR        "/data/run01/scxj523/zsh/project/AIForCompression/models/DCMVC/src/build/3rdparty/ryg_rans/ryg_rans-src"
        BINARY_DIR        "/data/run01/scxj523/zsh/project/AIForCompression/models/DCMVC/src/build/3rdparty/ryg_rans/ryg_rans-build"
        DOWNLOAD_COMMAND  ""
        UPDATE_COMMAND    ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
else()
    ExternalProject_Add(ryg_rans
        GIT_REPOSITORY    https://github.com/rygorous/ryg_rans.git
        GIT_TAG           c9d162d996fd600315af9ae8eb89d832576cb32d
        GIT_SHALLOW       1
        SOURCE_DIR        "/data/run01/scxj523/zsh/project/AIForCompression/models/DCMVC/src/build/3rdparty/ryg_rans/ryg_rans-src"
        BINARY_DIR        "/data/run01/scxj523/zsh/project/AIForCompression/models/DCMVC/src/build/3rdparty/ryg_rans/ryg_rans-build"
        UPDATE_COMMAND    ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
endif()
