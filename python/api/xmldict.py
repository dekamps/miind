"""
This module is taken from the xmljson package

"""
from collections import OrderedDict, Counter
from xml.etree.ElementTree import Element


def _fromstring(value):
   '''Convert XML string value to None, boolean, int or float'''
   if not value:
       return None
   std_value = value.strip().lower()
   if std_value == 'true':
       return True
   elif std_value == 'false':
       return False
   try:
       return int(std_value)
   except ValueError:
       pass
   try:
       return float(std_value)
   except ValueError:
       pass
   return value


def _tostring(value):
    '''Convert value to XML compatible string'''
    if value is True:
        value = 'TRUE'
    elif value is False:
        value = 'FALSE'
    return unicode(value)


def xml_to_dict(root, text_content='content', dict_type=OrderedDict):
    '''Convert etree.Element into a dictionary'''
    value = dict_type()
    children = [node for node in root if isinstance(node.tag, basestring)]
    for attr, attrval in root.attrib.items():
        value[attr] = _fromstring(attrval)
    if root.text:
        text = root.text.strip()
        if text and text_content is not None:
            value[text_content] = _fromstring(text)
        elif text and text_content is None:
            value = _fromstring(text)
    count = Counter(child.tag for child in children)
    for child in children:
        if count[child.tag] == 1:
           value.update(xml_to_dict(child, text_content=text_content,
                                    dict_type=dict_type))
        else:
            result = value.setdefault(child.tag, list())
            result += xml_to_dict(child, text_content=text_content,
                                  dict_type=dict_type).values()
    return dict_type([(root.tag, value)])


def dict_to_xml(data, root=None, text_content='content'):
    '''Convert data structure into a list of etree.Element'''
    result = list() if root is None else root
    if isinstance(data, dict):
        for key, value in data.items():
            value_is_list = isinstance(value, list)
            value_is_dict = isinstance(value, dict)
            # Add attributes and text to result (if root)
            if root is not None:
                # Handle text content
                if key == text_content:
                    result.text = _tostring(value)
                    continue
                # Treat scalars as text content, not children
                if not value_is_dict and not value_is_list:
                    result.set(key, _tostring(value))
                    continue
            # Add other keys as one or more children
            values = value if value_is_list else [value]
            for value in values:
                elem = Element(key)
                result.append(elem)
                dict_to_xml(value, root=elem, text_content=text_content)
    else:
        if text_content is None and root is not None:
            root.text = _tostring(data)
        else:
            result.append(Element(_tostring(data)))
    return result
