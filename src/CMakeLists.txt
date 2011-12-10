cmake_minimum_required(VERSION 2.8)

ADD_EXECUTABLE(main main.cpp filereader.cpp recognition.cpp svm-predict.cpp svm.cpp textlocation.cpp utility.cpp)

FIND_PACKAGE(OpenCV REQUIRED)
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})

FILE(COPY ${PROJECT_SOURCE_DIR}/rc/model		DESTINATION ${PROJECT_BINARY_DIR}/bin)
FILE(COPY ${PROJECT_SOURCE_DIR}/rc/scale_para	DESTINATION ${PROJECT_BINARY_DIR}/bin)
FILE(COPY ${PROJECT_SOURCE_DIR}/rc/test.jpg		DESTINATION ${PROJECT_BINARY_DIR}/bin)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

