#
# configure/build miind mpi tests
#
##############################################################################
# check if have valgrind
##############################################################################
SET(HAVE_VALGRIND TRUE)
FIND_PROGRAM(VALGRIND_EXECUTABLE valgrind)
IF(VALGRIND_EXECUTABLE STREQUAL "VALGRIND_EXECUTABLE-NOTFOUND")
    MESSAGE(STATUS "WARNING: Could not find valgrind. Will NOT build memory tests.")
    SET(HAVE_VALGRIND FALSE)
ENDIF(VALGRIND_EXECUTABLE STREQUAL "VALGRIND_EXECUTABLE-NOTFOUND")

##############################################################################
# check if MPI Executable is available
##############################################################################
SET(HAVE_MPIRUN TRUE)
IF(MPIEXEC STREQUAL "MPIEXEC-NOTFOUND")
    MESSAGE(FATAL_ERROR "ERROR: Could not find mpirun. Will NOT build tests. Either deactivate test or provide the variable MPIEXEC")
    SET(HAVE_MPIRUN FALSE)
ENDIF(MPIEXEC STREQUAL "MPIEXEC-NOTFOUND")

##############################################################################
# macro definitions
##############################################################################
MACRO(ADD_MIIND_MPI_TEST name exe src)
    STRING(REGEX REPLACE "test_([^ ]+).*" "\\1" test "${exe}" )
    # build the test
    ADD_EXECUTABLE(${exe} ${src})
    
    SET(TEST_LIBS  
        ${MIIND_LIBRARY_PREFIX}mpi      
        ${MPI_LIBRARIES} 
        ${Boost_LIBRARIES}
    )

    TARGET_LINK_LIBRARIES(${exe} ${TEST_LIBS})

    
    
    ADD_TEST(${name} ${MPIEXEC} -np 2 ${exe})
    MESSAGE(STATUS "Adding test for ${name}: ${exe}.")
    # add target for the test
    STRING(REGEX REPLACE "test_([^ ]+).*" "unit_\\1" unittest_target "${exe}" )
    ADD_CUSTOM_TARGET(${unittest_target} COMMAND mpirun -np 2 ${CMAKE_CURRENT_BINARY_DIR}/${exe})
    
    IF (HAVE_VALGRIND) 
        #add memory test using valgrind
        STRING(REGEX REPLACE "test_([^ ]+).*" "mem_\\1" memtest_name "${exe}" )
        LIST(APPEND memtest_names ${memtest_name})
        LIST(APPEND exe_names ${exe})
        #Add target for the memory test
        ADD_CUSTOM_TARGET(${memtest_name} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/memcheck.py ${CMAKE_CURRENT_BINARY_DIR}/${exe})
    ENDIF (HAVE_VALGRIND) 
ENDMACRO(ADD_MIIND_MPI_TEST name exe src)

##############################################################################
# include directories
##############################################################################
INCLUDE_DIRECTORIES( 
    ../../MPILib/include
    ../../MPILib/include/utilities 
)




##############################################################################
# Add here the test
##############################################################################
ADD_MIIND_MPI_TEST("MPINetwork" test_MPINetwork_test MPINetwork_test.cpp)
ADD_MIIND_MPI_TEST("utilities/CircularDistrbution" test_CircularDistribution_test utilities/CircularDistribution_test.cpp)
ADD_MIIND_MPI_TEST("utilities/Exception" test_Exception_test utilities/Exception_test.cpp)
ADD_MIIND_MPI_TEST("utilities/ParallelException" test_ParallelException_test utilities/ParallelException_test.cpp)

#ADD_MIIND_MPI_TEST("Example" test_example_test example_test.cpp)


##############################################################################
# generation of the memtests
##############################################################################
LIST(LENGTH memtest_names numtests)
IF(numtests GREATER 0)
    MATH(EXPR numtests ${numtests}-1)
    FOREACH(i RANGE 0 ${numtests})
        LIST(GET memtest_names ${i} memtest_name)
        LIST(GET exe_names ${i} exe)
        MESSAGE(STATUS "Adding memory test for ${memtest_name}: ${exe}.")
        ADD_TEST(${memtest_name}
            ${MPIEXEC} -np 1 ${CMAKE_CURRENT_SOURCE_DIR}/memtest.py ${CMAKE_CURRENT_BINARY_DIR}/${exe} ${CMAKE_BINARY_DIR})
    ENDFOREACH(i RANGE 0 ${numtests}-1)
ENDIF(numtests GREATER 0)

