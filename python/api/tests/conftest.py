import pytest
import os
from miindio import MiindIO


xml_path = os.path.join(os.path.dirname(__file__), 'testme.xml')
test_path = xml_path.replace('.xml', '') + '_temp'


def pytest_namespace():
    return {"XML_PATH": xml_path,
            "PATH": test_path}


@pytest.fixture(scope='module')
def io_run():
    io = MiindIO(xml_path)
    if not io.run_exists:
        io.submit()
        io.run()
    return io
