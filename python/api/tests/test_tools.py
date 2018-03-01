import pytest
import os
import json
import copy
import api
import xml.etree.ElementTree as ET
from api.tools import set_params, convert_xml_dict, dict_changed

def test_dictify():
    from api.tools import dictify
    a = {
        'a': {'b': 'c'},
        'd': ['e', 'f', [1, 2, 'd']],
        'g': 1,
        'h': 'j'
    }
    b = copy.deepcopy(a)
    dictify(a)
    val = {0: 'e', 1: 'f', 2: {0: 1, 1: 2, 2: 'd'}}
    assert a['d'] == val
    assert dict_changed(b, a) == set('d')


def test_listify():
    from api.tools import listify
    a = {
        'a': {'b': 'c'},
        'd': {0: 'e', 1: 'f', 2: {0: 1, 1: 2, 2: 'd'}},
        'g': 1,
        'h': 'j'
    }
    b = copy.deepcopy(a)
    listify(a)
    assert a['d'] == ['e', 'f', [1, 2, 'd']]


def test_read_write_dict_equal():
    from api.tools import dump_xml, to_json
    p = convert_xml_dict(pytest.XML_PATH)
    to_json(p, pytest.PATH + '.json')
    p = json.loads(json.dumps(p))
    dump_xml(p, pytest.PATH + '.xml')
    q = convert_xml_dict(pytest.PATH + '.xml')
    q = json.loads(json.dumps(q))
    assert p == q


# def test_read_write_string_equal():
#     from miindio.tools import convert_xml_dict, dump_xml, to_json
#     p = convert_xml_dict(pytest.XML_PATH)
#     dump_xml(p, pytest.PATH + '.xml')
#     q_string = ET.tostring(ET.parse(pytest.PATH + '.xml').getroot())
#     p_string = ET.tostring(ET.parse(pytest.XML_PATH).getroot())
#     assert q_string == p_string


def test_set_value_no_content():
    params = {'t_end': .01}
    p = convert_xml_dict(pytest.XML_PATH)
    q = copy.deepcopy(p)
    set_params(q, **params)
    assert p == q
    set_params(p, t_end=1)
    assert p['Simulation']['SimulationRunParameter']['t_end']['content'] == 1

def test_set_value_no_content_deep():
    params = {
        'Algorithm': {
            2: {'expression': 1.}
        }
    }
    p = convert_xml_dict(pytest.XML_PATH)
    set_params(p, **params)
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 1.
    assert base['type'] == "RateFunctor"


def test_set_no_content_deep_string():
    p = convert_xml_dict(pytest.XML_PATH)
    set_params(p, Algorithm='2/expression/1.')
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 1.
    assert base['type'] == "RateFunctor"


def test_change_params_dict():
    params = {
        'Connection': {
            3: {'content': '-1000 100. 0001'}
        }
    }
    p = convert_xml_dict(pytest.XML_PATH)
    set_params(p, **params)
    base = p['Simulation']['Connections']['Connection'][3]
    assert base['content'] == '-1000 100. 0001'
    assert base['In'] == 'adex E'
    assert base['Out'] == 'adex I'


def test_change_params_dict2():
    params = {
        'Algorithm': {
            2: {'expression': {'content': 1.}}
        }
    }
    p = convert_xml_dict(pytest.XML_PATH)
    set_params(p, **params)
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 1.
    assert base['type'] == "RateFunctor"


def test_change_params_string():
    p = convert_xml_dict(pytest.XML_PATH)
    set_params(p, Algorithm='2/expression/content/1.')
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 1.
    assert base['type'] == "RateFunctor"


def test_set_val_new_type():
    p = convert_xml_dict(pytest.XML_PATH)
    with pytest.raises(TypeError):
        set_params(p, Algorithm='2/expression/1')


def test_set_val_new_structure():
    p = convert_xml_dict(pytest.XML_PATH)
    with pytest.raises(TypeError):
        set_params(p, Algorithm='2/1')


def test_set_attr():
    p = convert_xml_dict(pytest.XML_PATH)
    set_params(p, Node='0/algorithm/yoyo')
    assert p['Simulation']['Nodes']['Node'][0]['algorithm'] == 'yoyo'


def test_set_multiple_values():
    from miindio.tools import dump_xml
    params = {'Algorithm': '2/expression/2000.',
              'Connection': {2: '50 -1 1',
                             3: '200 1 1'}}
    p = convert_xml_dict(pytest.XML_PATH)
    set_params(p, **params)
    dump_xml(p, pytest.PATH + '.xml')
    base = p['Simulation']['Algorithms']['Algorithm'][2]
    assert base['expression']['content'] == 2000.
