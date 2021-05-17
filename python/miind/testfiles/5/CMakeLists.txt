set( TEST_FILES
lif.model
lif.xml
lif_0.01_0_0_0_.mat
)

set(TESTFILES_BASE_DIR share/miind/python/miind/testfiles)

install( FILES ${TEST_FILES}
    DESTINATION ${TESTFILES_BASE_DIR}/5 )