cmake_minimum_required(VERSION 3.10)

set(LIB sgm_cuda)

# Source files
file(GLOB SOURCES "*.cu")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS} ${CUDA_LIBRARY_DIRS})

set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3
  -gencode=arch=compute_30,code=sm_30
  -gencode=arch=compute_35,code=sm_35
  -gencode=arch=compute_50,code=sm_50
  -gencode=arch=compute_52,code=sm_52
  -gencode=arch=compute_61,code=sm_61
)

# Generate executable and link
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
cuda_add_library (${LIB} STATIC ${SOURCES})
target_include_directories(${LIB} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(${LIB} ${OpenCV_LIBS} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
