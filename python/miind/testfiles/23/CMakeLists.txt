set( TEST_FILES
lif.model
lif.xml
lif.fid
)

set(TESTFILES_BASE_DIR share/miind/python/miind/testfiles)

install( FILES ${TEST_FILES}
    DESTINATION ${TESTFILES_BASE_DIR}/23 )