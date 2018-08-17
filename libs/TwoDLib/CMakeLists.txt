
project( ${MIIND_LIBRARY_PREFIX}twod )

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
CSRMatrix.cpp			
Master.cpp			
TransitionMatrix.cpp	 
Cell.cpp											
MasterOMP.cpp		   
MasterOdeint.cpp		   
QuadGenerator.cpp  
TransitionMatrixGenerator.cpp 
pugixml.cpp
CheckSystem.cpp														   		      
Mesh.cpp			      
Quadrilateral.cpp  
Triangle.cpp
kd.cpp
Fid.cpp		   				      			  
Euler.cpp
MeshTree.cpp   
Redistribution.cpp  
TriangleGenerator.cpp
LineSegment.cpp												  		 		     
Ode2DSystem.cpp   
Ode2DSystemGroup.cpp   
Stat.cpp   
Uniform.cpp 
)

set( TWOD_HEADERS
Bin.hpp
InputCombination.hpp		
MeshAlgorithmCode.hpp		
Redistribution.hpp	 
TwoDLibException.hpp
CSRMatrix.hpp											
LineSegment.hpp		  
MeshTree.hpp   
Stat.hpp   
TwoDLib.hpp
Euler.hpp
Cell.hpp														   		  
Master.hpp   
MasterOdeint.hpp   
Ode2DSystem.hpp   
Ode2DSystemGroup.hpp   
TransitionList.hpp  
Uniform.hpp
CheckSystem.hpp																  	       			 
MasterOMP.hpp	       
TransitionMatrix.hpp  
modulo.hpp
Coordinates.hpp										 		       			 
MasterParameter.hpp    
Point.hpp   
TransitionMatrixGenerator.hpp 
pugiconfig.hpp
Fid.hpp																		 				
Mesh.hpp   
QuadGenerator.hpp  
Triangle.hpp   
pugixml.hpp
FiducialElement.hpp																	  	     
MeshAlgorithm.hpp		       
Quadrilateral.hpp  
TriangleGenerator.hpp
)


add_library( ${MIIND_LIBRARY_PREFIX}twod ${LIB_TYPE} ${TWOD_SRC} ${TWOD_HEADERS})

target_link_libraries( ${MIIND_LIBRARY_PREFIX}twod
    ${gsl_LINK_FLAGS} )

install( FILES ${numtools_HEADERS}
    DESTINATION include/miind/TwoDLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}twod
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
