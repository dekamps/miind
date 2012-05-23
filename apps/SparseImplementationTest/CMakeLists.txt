


add_executable( sparseimplementationtest main.cpp )

include_directories( ${GSL_INCLUDE_DIRS} )
target_link_libraries( sparseimplementationtest ${MIIND_LIBRARY_PREFIX}sparseimplementation )

