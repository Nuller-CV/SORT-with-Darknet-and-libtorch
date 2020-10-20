cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(darknet-sort)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#link for opencv
set(OpenCV_DIR "/home/nuller-cv/soft/opencv4_4/opencv-4.4.0/")
find_package(OpenCV REQUIRED )


# link for libtorch
set(Torch_DIR /home/nuller-cv/soft/libtorch/share/cmake/Torch/)
find_package(Torch REQUIRED)
# target_link_libraries("${PROJECT_NAME}" "${TORCH_LIBRARIES}")

# For profiler on Ubuntu
if (CMAKE_BUILD_TYPE STREQUAL Debug AND NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif ()

# Output directory structure
set(OUTPUT_DIR "result")
set(TARGETS_DIR_NAME "targets")
set(TRAJ_TXT_NAME "trajectories.txt")
set(SNAPSHOTS_DIR_NAME "snapshots")
set(VIDEO_NAME "compressed.flv")
configure_file(
        "${PROJECT_SOURCE_DIR}/config.h.in"
        "${PROJECT_BINARY_DIR}/config.h"
)

# GCC need to link against stdc++fs
if(MSVC)
set(STDCXXFS "")
else()
set(STDCXXFS "stdc++fs")
endif()

# .exe and .dll
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


add_subdirectory(tracking)
# add_subdirectory(door)
add_subdirectory(processing)
# add_subdirectory(GUI)
