# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/129/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/129/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort

# Include any dependencies generated for this target.
include processing/CMakeFiles/processing.dir/depend.make

# Include the progress variables for this target.
include processing/CMakeFiles/processing.dir/progress.make

# Include the compile flags for this target's objects.
include processing/CMakeFiles/processing.dir/flags.make

processing/CMakeFiles/processing.dir/main.cpp.o: processing/CMakeFiles/processing.dir/flags.make
processing/CMakeFiles/processing.dir/main.cpp.o: processing/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object processing/CMakeFiles/processing.dir/main.cpp.o"
	cd /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/processing.dir/main.cpp.o -c /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing/main.cpp

processing/CMakeFiles/processing.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/processing.dir/main.cpp.i"
	cd /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing/main.cpp > CMakeFiles/processing.dir/main.cpp.i

processing/CMakeFiles/processing.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/processing.dir/main.cpp.s"
	cd /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing/main.cpp -o CMakeFiles/processing.dir/main.cpp.s

# Object files for target processing
processing_OBJECTS = \
"CMakeFiles/processing.dir/main.cpp.o"

# External object files for target processing
processing_EXTERNAL_OBJECTS =

bin/processing: processing/CMakeFiles/processing.dir/main.cpp.o
bin/processing: processing/CMakeFiles/processing.dir/build.make
bin/processing: /home/nuller-cv/soft/darknet/libdarknet.so
bin/processing: tracking/libtracking.so
bin/processing: /usr/local/lib/libopencv_dnn.so.4.4.0
bin/processing: /usr/local/lib/libopencv_gapi.so.4.4.0
bin/processing: /usr/local/lib/libopencv_highgui.so.4.4.0
bin/processing: /usr/local/lib/libopencv_ml.so.4.4.0
bin/processing: /usr/local/lib/libopencv_objdetect.so.4.4.0
bin/processing: /usr/local/lib/libopencv_photo.so.4.4.0
bin/processing: /usr/local/lib/libopencv_stitching.so.4.4.0
bin/processing: /usr/local/lib/libopencv_video.so.4.4.0
bin/processing: /usr/local/lib/libopencv_calib3d.so.4.4.0
bin/processing: /usr/local/lib/libopencv_features2d.so.4.4.0
bin/processing: /usr/local/lib/libopencv_flann.so.4.4.0
bin/processing: /usr/local/lib/libopencv_videoio.so.4.4.0
bin/processing: /usr/local/lib/libopencv_imgcodecs.so.4.4.0
bin/processing: /usr/local/lib/libopencv_imgproc.so.4.4.0
bin/processing: /usr/local/lib/libopencv_core.so.4.4.0
bin/processing: processing/CMakeFiles/processing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/processing"
	cd /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/processing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
processing/CMakeFiles/processing.dir/build: bin/processing

.PHONY : processing/CMakeFiles/processing.dir/build

processing/CMakeFiles/processing.dir/clean:
	cd /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing && $(CMAKE_COMMAND) -P CMakeFiles/processing.dir/cmake_clean.cmake
.PHONY : processing/CMakeFiles/processing.dir/clean

processing/CMakeFiles/processing.dir/depend:
	cd /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing /home/nuller-cv/Program/ar_hub_test/weixu_reimplement_SORT/darknet_sort/processing/CMakeFiles/processing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : processing/CMakeFiles/processing.dir/depend

