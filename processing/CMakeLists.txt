#opencv
add_definitions(-std=c++11)
ADD_DEFINITIONS(-DOPENCV)
ADD_DEFINITIONS(-DGPU)

#########   opencv      #########
set(OpenCV_DIR "/home/nuller-cv/soft/opencv4_4/opencv-4.4.0/")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
#########   darknet     #########
include_directories(/home/nuller-cv/soft/darknet/include/)
find_library(darknet libdarknet.so /home/nuller-cv/soft/darknet/)
add_executable(processing "main.cpp")
target_link_libraries(processing ${OpenCV_LIBS}  ${darknet})

aux_source_directory(. PROCESSING_SRCS)

target_link_libraries(processing ${OpenCV_LIBS} tracking ${STDCXXFS})
target_include_directories(processing PRIVATE "${PROJECT_BINARY_DIR}")