cmake_minimum_required(VERSION 3.10)

PROJECT(demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_subdirectory(camera)
add_subdirectory(sgm_cuda)

cuda_add_executable(${PROJECT_NAME} "main.cu")

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ocams_camera sgm_cuda)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/camera/ocams_calibration_720p.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
