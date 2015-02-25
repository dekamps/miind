
project( ${MIIND_LIBRARY_PREFIX}util )

if ( WIN32 )
set (LIB_TYPE STATIC)
else( WIN32 ) 
set (LIB_TYPE SHARED)
endif (WIN32)

set( util_SRC
	AttributeList.cpp
    GeneralException.cpp
    Incremental.cpp
    Line.cpp
    LineAttributes.cpp
    LogStream.cpp
    Named.cpp
    ParameterScan.cpp
    Persistant.cpp
    PositionInCm.cpp
    RGBValue.cpp
    Rect.cpp
    RectAttributes.cpp
    Sequence.cpp
    SequenceFactory.cpp
    SequenceIterator.cpp
    SequenceIteratorIterator.cpp
    Series.cpp
    Streamable.cpp
    Timer.cpp
    UtilLibException.cpp
    UtilTest.cpp )
    
set( util_HEADERS
    AbstractException.h
	AttributeList.h
    BasicDefinitions.h
    ColorTable.h
    ConcreteStreamable.h
    FailAlloc.h
    GeneralException.h
    Incremental.h
    IsFinite.h
    IsNan.h
    Line.h
    LineAttributes.h
    LocalDefinitions.h
    LogStream.h
    MToCm.h
	Named.h
    NumericException.h
    PaperFormatError.h
    ParameterScan.h
	Persistant.h
    PositionInCm.h
    Rect.h
    RectAttributes.h
    RGBValue.h
    Series.h
    Sequence.h
    SequenceFactory.h
    SequenceIterator.h
    SequenceIteratorIterator.h
    Shape.h
    SizeInCm.h
    Streamable.h
    TimeException.h
    Timer.h
    ToValarray.h
    ToVector.h
    UtilLibException.h
    UtilLib.h
    UtilTest.h )

add_library( ${MIIND_LIBRARY_PREFIX}util ${LIB_TYPE} ${util_SRC} ${util_HEADERS})

install( FILES ${util_HEADERS}
    DESTINATION include/miind/UtilLib )


install( TARGETS ${MIIND_LIBRARY_PREFIX}util
    LIBRARY DESTINATION lib 
    ARCHIVE DESTINATION lib )
