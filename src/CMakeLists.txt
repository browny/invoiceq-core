cmake_minimum_required(VERSION 2.8)

ADD_EXECUTABLE(main main.cpp filereader.cpp recognition.cpp svm-predict.cpp svm.cpp textlocation.cpp utility.cpp)

INCLUDE_DIRECTORIES(../include/)
INCLUDE_DIRECTORIES(../include/opencv/)

TARGET_LINK_LIBRARIES(
	main 
	${PROJECT_SOURCE_DIR}/lib/libopencv_core.so.2.3.1
	${PROJECT_SOURCE_DIR}/lib/libopencv_highgui.so.2.3.1
	${PROJECT_SOURCE_DIR}/lib/libopencv_imgproc.so.2.3.1
)

file(COPY ${PROJECT_SOURCE_DIR}/rc/model		DESTINATION ${PROJECT_BINARY_DIR}/bin)
file(COPY ${PROJECT_SOURCE_DIR}/rc/scale_para	DESTINATION ${PROJECT_BINARY_DIR}/bin)
file(COPY ${PROJECT_SOURCE_DIR}/rc/test.jpg		DESTINATION ${PROJECT_BINARY_DIR}/bin)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

MESSAGE(STATUS "The executable file will be in" ${PROJECT_BINARY_DIR}/bin)

