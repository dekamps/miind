import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from pprint import pprint
import json
import os
import os.path as op
import numpy as np
import copy
from collections import Mapping, OrderedDict
from xmldict import dict_to_xml, xml_to_dict, _fromstring
import directories


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def getMiindBuildPath():
    return os.path.join(directories.miind_root(), 'build')

def getMiindPythonPath():
    return os.path.join(directories.miind_root(), 'python')

def getMiindAppsPath():
    build_path = getMiindBuildPath()
    return op.join(build_path, 'apps')

def split_fname(fname, ext):
    fname = op.split(fname)[1]
    if not ext.startswith('.'):
        ext = '.' + ext
    if fname.endswith(ext):
        modelname = op.splitext(fname)[0]
        modelfname = fname
    else:
        modelname = fname
        modelfname = fname + ext
    return modelname, modelfname

def dict_changed(d1, d2):
    k1, k2 = [set(d.keys()) for d in (d1, d2)]
    intersect = k1.intersection(k2)
    return set(o for o in intersect if d2[o] != d1[o])


def prettify_xml(elem):
    """Return a pretty-printed XML string for an Element, string or dict.
    """
    if isinstance(elem, Mapping):
        string = convert_dict_xml(elem)
    elif isinstance(elem, ET.ElementTree):
        string = ET.tostring(elem, 'utf-8')
    elif isinstance(elem, str):
        string = elem
    else:
        raise TypeError('type {} not recognized.'.format(type(elem)))
    reparsed = minidom.parseString(string)
    return reparsed.toprettyxml(indent="\t")


def dump_xml(params, fpathout):
    path, fname = os.path.split(fpathout)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(fpathout, 'w') as f:
        f.write(prettify_xml(params))


def deep_update(d, other, strict=False):
    for k, v in other.items():
        d_v = d.get(k)
        if isinstance(v, Mapping) and isinstance(d_v, Mapping):
            deep_update(d_v, v, strict=strict)
        else:
            if strict:
                if d_v is None:
                    raise ValueError('Unable to update "{}" '.format(k) +
                                     'according to the "strict" rule.')
                if not type(d[k]) == type(v):
                    raise TypeError("Can't set type '{}' ".format(type(d[k])) +
                                    "to new type '{}' ".format(type(v)) +
                                    "according to the 'strict' rule.")
            d[k] = copy.deepcopy(v)


def dictify(d):
    assert isinstance(d, Mapping)
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = OrderedDict((i, ii) for i, ii in enumerate(v))
        if isinstance(d[k], Mapping):
            d[k] = dictify(d[k])
    return d


def listify(d):
    if isdictlist(d):
        d = [d[k] for k in sorted(d.keys())]
    if isinstance(d, Mapping):
        for key, val in d.items():
            d[key] = listify(val)
    if isinstance(d, list):
        for i, c in enumerate(d):
            d[i] = listify(c)
    return d


def map_key(dic, key, path=None):
    path = path if path is not None else []
    if isinstance(dic, Mapping):
        for k, v in dic.items():
            local_path = path[:]
            local_path.append(k)
            for b in map_key(v, key, local_path):
                 yield b
    if len(path) == 0:
        pass
    elif path[-1] == key:
        yield path, dic


def map_dict(dic, path=None):
    path = path if path is not None else []
    if isinstance(dic, Mapping):
        for k, v in dic.items():
            local_path = path[:]
            local_path.append(k)
            for b in map_dict(v, local_path):
                 yield b
    else:
        yield path, dic


def pack_list_dict(parts):
    if len(parts) == 1:
        return parts[0]
    elif len(parts) > 1:
        return {parts[0]: pack_list_dict(parts[1:])}
    return parts


def ispath(val):
    tpath = os.path.abspath(os.path.expanduser(val))
    if not os.path.exists(tpath) and not os.path.isdir(tpath):
        return False
    else:
        return True


def get_from_dict(dct, lst):
    from functools import reduce
    import operator
    return reduce(operator.getitem, lst, dct)


def set_params(params, **kwargs):
    if not kwargs:
        return
    dictify(params)
    for key, val in kwargs.items():
        if isinstance(val, (str, unicode)):
            if '/' in val:
                # make sure its not a path
                if not ispath(val):
                    tval = [_fromstring(v) for v in val.split('/')]
                    val = pack_list_dict(tval)
        old_mapping = list(map_key(params, key))
        if len(old_mapping) == 0:
            raise ValueError('Unable to map instance of "{}", '.format(key))
        if len(old_mapping) > 1:
            raise ValueError('Found multiple instances of "{}", '.format(key) +
                             'mapping must be unique')
        path, old_val = old_mapping[0]
        if isinstance(val, Mapping):
            for val_path, val_val in map_dict(val):
                if not val_path[-1] == 'content':
                    old_deep_val = get_from_dict(old_val, val_path)
                    if isinstance(old_deep_val, Mapping):
                        if 'content' in old_deep_val:
                            val_path.append('content')
                val_path.append(val_val)
                deep_update(val, pack_list_dict(val_path))
        elif 'content' in old_val:
            old_val['content'] = val
            val = old_val
        path.append(val)
        packed_list = pack_list_dict(path)
        deep_update(params, packed_list, strict=True)
    listify(params)


def isdictlist(val):
    if isinstance(val, Mapping):
        if all(isinstance(k, int) for k in val.keys()):
            idxs = sorted(val.keys())
            if all(k2 == k1 + 1 for k1, k2 in zip(idxs, idxs[1:])):
                return True
    return False


def isnumeric(val):
    assert isinstance(val, unicode)
    return remove_txt(val, ' ', '-', '.').isnumeric()


def remove_txt(txt, *args):
    for arg in args:
        txt = txt.replace(arg, '')
    return txt


def convert_xml_dict(filename):
    root = ET.parse(filename).getroot()
    odict = xml_to_dict(root)
    return odict


def convert_dict_xml(dictionary):
    elem = dict_to_xml(dictionary)
    assert len(elem) == 1
    return ET.tostring(elem[0])


def pretty_print_params(arg):
    if isinstance(arg, str):
        arg = convert_xml_dict(arg)
    else:
        assert isinstance(arg, Mapping)
    pprint(json.loads(json.dumps(arg)))


def to_json(arg, fname='params.json'):
    if isinstance(arg, str):
        arg = convert_xml_dict(arg)
    else:
        assert isinstance(arg, Mapping)
    with open(fname, 'w') as outfile:
        json.dump(arg, outfile,
                  sort_keys=True, indent=4)


def extract_mesh(modelpath, overwrite=False):
    meshpath = modelpath.replace('.model', '.mesh.bak')
    if not os.path.exists(meshpath) or overwrite:
        print('No mesh, generates from ".model" file..')
        tree = ET.parse(modelpath)
        root = tree.getroot()
        for child in root:
            if child.tag == 'Mesh':
                meshstr=ET.tostring(child)
        with open(meshpath,'w') as fmesh:
            fmesh.write(meshstr)
    return meshpath
