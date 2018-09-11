project( ${MIIND_LIBRARY_PREFIX}cudatwod )

find_package(CUDA  REQUIRED)
link_directories( ${GSL_LIBRARY_DIRS} )
include( RequireGSL )
include_directories(${CUDA_INCLUDE_DIRS})
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
)
cuda_add_library( ${MIIND_LIBRARY_PREFIX}cudatwod ${LIB_TYPE} ${TWOD_SRC} ${TWOD_HEADERS} ${SRC} )
add_definitions( CMAKE_CUDA_FLAGS=”-arch=sm_35” )
target_compile_options( ${MIIND_LIBRARY_PREFIX}cudatwod PUBLIC -std=c++11)
target_link_libraries( ${MIIND_LIBRARY_PREFIX}cudatwod
    ${gsl_LINK_FLAGS} )

install( FILES ${numtools_HEADERS}
    DESTINATION include/miind/CudaTwoDLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}cudatwod
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
