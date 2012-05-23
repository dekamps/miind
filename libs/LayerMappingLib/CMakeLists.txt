ADD_SUBDIRECTORY( Swig )

project( ${MIIND_LIBRARY_PREFIX}layermapping )


if( WIN32 )
  find_package( WINGSL REQUIRED )
  if( NOT GSL_FOUND )
    message( FATAL_ERROR "WinGsl not found (module ConnectionismLib)." )
  endif( NOT GSL_FOUND )
  set( LIB_TYPE STATIC )
else( WIN32 )
  include( RequireGSL )
  set( LIB_TYPE SHARED )  
endif( WIN32 )

include_directories( ${GSL_INCLUDE_DIRS} )
link_directories( ${GSL_LIBRARY_DIRS} )


link_directories(
	${FFTW_LIBRARY_DIRS}  )
include_directories(
	${FFTW_INCLUDE_DIRS}  )

if( GTKMM_FOUND )
	add_definitions( -DHAVE_GTKMM )
	ADD_SUBDIRECTORY( gtkmm )
endif( GTKMM_FOUND )

if( FFTW_FOUND )
	add_definitions( -DHAVE_FFTW )
endif( FFTW_FOUND )

set( layermapping_SRC
# 	HMAXInterface.cpp
	Models.cpp
	Exception.cpp
	Test.cpp
	Util.cpp )

set( layermapping_HEADERS
	./algorithm.h
	./ReceptiveFieldIteratorCode.h
	./FunctionFactory.h
	./FeatureMapNetworkCode.h
	./FeatureMapIterator.h
	./Functions/Mean.h
	./Functions/MaxCode.h
	./Functions/MinCode.h
	./Functions/MeanCode.h
	./Functions/ProductCode.h
	./Functions/SumCode.h
	./Functions/Product.h
	./Functions/Sum.h
	./Functions/Min.h
	./Functions/Convolution.h
	./Functions/ConvolutionCode.h
	./Functions/Max.h
	./Functions/CompositeFeature.h
	./Functions/CompositeFeatureCode.h
	./Functions/Perceptron.h
	./Functions/PerceptronCode.h
	./Functions/ArgMax.h
	./Functions/ArgMaxCode.h
	./Functions/Identity.h
	./Functions/IdentityCode.h
	./Functions/FFTC.h
	./Functions/FFTCCode.h
	./Functions/FFTR.h
	./Functions/FFTRCode.h
	./Functions/Combine.h
	./Functions/CombineCode.h
	./Functions/Scale.h
	./Functions/ScaleCode.h
	./Functions/StandardDeviation.h
	./Functions/StandardDeviationCode.h
	./Functions/SimpleCellResponse.h
	./Functions/SimpleCellResponseCode.h
	./Functions/ProductComplexImag.h
	./Functions/ProductComplexImagCode.h
	./Functions/ProductComplexReal.h
	./Functions/ProductComplexRealCode.h
	./FeatureMapNetwork.h
	./NetworkEnsemble.h
	./NetworkEnsembleCode.h
	./NetworkInterface.h
	./NetworkInterfaceCode.h
	./FeatureMapNode.h
	./FunctionFactoryCode.h
	./AbstractFunction.h
	./ReceptiveFieldIterator.h
	./FeatureMap.h
	./FeatureMapIteratorCode.h
	./FeatureMapCode.h
	./FeatureMapNodeCode.h
	./Exception.h
	./Util.h
# 	./HMAXInterface.h
	./LayerMappingLib.h
	./ReceptiveField.h
	./ReceptiveFieldCode.h
	./Test.h
	./Models.h )

add_library( ${MIIND_LIBRARY_PREFIX}layermapping ${LIB_TYPE} ${layermapping_SRC} ${layermapping_HEADERS} )

target_link_libraries( ${MIIND_LIBRARY_PREFIX}layermapping ${FFTW_LIBRARIES} )


install( FILES ${layermapping_HEADERS}
    DESTINATION include/miind/LayerMappingLib )

install( TARGETS ${MIIND_LIBRARY_PREFIX}layermapping
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib )
