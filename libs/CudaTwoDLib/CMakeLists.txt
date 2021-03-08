project( ${MIIND_LIBRARY_PREFIX}cudatwod )

if (WIN32) # When cmake generates the VS files, it needs to know this is a cuda project
enable_language(CUDA)
endif (WIN32)

message ("************************************************************")
message ("************************************************************")
message ("************************************************************")

message ("CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER} CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")

message ("************************************************************")
message ("************************************************************")
message ("************************************************************")

find_package(CUDA  REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories( ${GSL_INCLUDE_DIRS} )
set( LIB_TYPE STATIC )

#one below that MPILib is included in the paths they are needed for the camke config files
INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../
   ${CMAKE_CURRENT_BINARY_DIR}/../
)

set( TWOD_SRC
CSRAdapter.cu
CudaOde2DSystemAdapter.cu
CudaEuler.cu
)

set( TWOD_HEADERS
CSRAdapter.cuh
CudaOde2DSystemAdapter.cuh
CudaEuler.cuh
CudaTwoDLib.hpp
)
cuda_add_library( ${MIIND_LIBRARY_PREFIX}cudatwod ${LIB_TYPE} ${TWOD_SRC} ${TWOD_HEADERS} ${SRC} )
target_compile_options( ${MIIND_LIBRARY_PREFIX}cudatwod PUBLIC )
target_link_libraries( ${MIIND_LIBRARY_PREFIX}cudatwod
    ${gsl_LINK_FLAGS} )


install( FILES ${TWOD_HEADERS}
    DESTINATION include/miind/CudaTwoDLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}cudatwod
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
