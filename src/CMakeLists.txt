cmake_minimum_required(VERSION 2.8)

ADD_EXECUTABLE(main main.cpp filereader.cpp recognition.cpp svm-predict.cpp svm.cpp textlocation.cpp utility.cpp)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	# Mac OS X specific code
	FIND_LIBRARY(CORE_LIB    NAMES opencv_core    PATHS "../lib/")
	FIND_LIBRARY(IMGPROC_LIB NAMES opencv_imgproc PATHS "../lib/")
	FIND_LIBRARY(HIGHGUI_LIB NAMES opencv_highgui PATHS "../lib/")
	TARGET_LINK_LIBRARIES(main ${CORE_LIB})
	TARGET_LINK_LIBRARIES(main ${IMGPROC_LIB})
	TARGET_LINK_LIBRARIES(main ${HIGHGUI_LIB})
	SET(OperatingSystem "Mac OS X")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	# Linux specific code
	SET(OperatingSystem "Linux")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

FILE(COPY ${PROJECT_SOURCE_DIR}/rc/model		DESTINATION ${PROJECT_BINARY_DIR}/bin)
FILE(COPY ${PROJECT_SOURCE_DIR}/rc/scale_para	DESTINATION ${PROJECT_BINARY_DIR}/bin)
FILE(COPY ${PROJECT_SOURCE_DIR}/rc/test.jpg		DESTINATION ${PROJECT_BINARY_DIR}/bin)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

MESSAGE(STATUS "The executable file will be in" ${PROJECT_BINARY_DIR}/bin)

