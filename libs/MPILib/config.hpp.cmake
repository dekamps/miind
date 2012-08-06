
#ifndef MPILIB_CONFIG_HPP_
#define MPILIB_CONFIG_HPP_

#cmakedefine ENABLE_MPI
#cmakedefine DEBUG

#ifdef DEBUG
    #undef NDEBUG
#else
    #define NDEBUG
#endif


#endif //MPILIB_CONFIG_HPP_