import string
import xml.etree.ElementTree as ET

class xml_tag_convertor:
    '''Take an xml tag directly from a simulation XML file.
    Convert it into a tag/dictonary tuple'''

    def __init__(self, tag):
        ''' This tag is a tag line from the XML file, whose values are candidates
        for replacement. This tag line can be convereted into a tag, attribute tuple which
        is the internal representation used by XML file to modify tags.'''
        self.tree = ET.fromstring(tag)


    def convert(self):
        return  self.tree.tag, self.tree.attrib

class xml_tag:

    def __init__(self,tag):
        self.convertor = xml_tag_convertor(tag)
        self.name = self.convertor.convert()[0]
        self.dict = self.convertor.convert()[1]


class xml_file:
    '''Represents an XML file as a parsed tree. One can present an XML tag in
    tag, attribute format, and if the tag can be identified within the file,
    the attribute values in the file are replaced by those in the presented tag.'''

    def __init__(self,file_name):
        '''Initialize with the XML file name'''
        self.name = file_name
        self.tree = ET.parse(file_name)


    def get_text_attribute(self, xml_tag, split):
        '''Returns a list of text values for a given tag and dictionary values. In a tag such as <Variable Name="harry">42</Variable>, the dictory value is Name:harry.
        Tags with a name not equal to "harry" will be considered different and would not be returned. The number of elements in the list is equal to the number of times the
        tags occur in the text regardless of where they occur. If split == True then each text is split in white spaces. For example, if <expression>a b c</expression>
        and later <expression>p q r</expression> occurs, and no others, then  [[ a, b, c], [p, q, r]] will be returned.'''

        root = self.tree.getroot()
        gen=root.iter(xml_tag.name)
        hits = [l for l in gen  if l.attrib == xml_tag.dict]

        if split == True:
            return [ x.text.split() for x in hits]
        else:
            return [ [x] for x in hits ] # always return a list of lists

    def join(self,l):
        ''' Joins text elements of list l with white space, returns the resulting string.'''
        s = ''
        for el in l[:-1]:
            s += el
            s += ' '
        s += l[-1]

        return s

    def insert_text_attribute(self, xml_tag, text_list, order):
        root = self.tree.getroot()
        l=root.iter(xml_tag.name)

        hits = []

        for el in l:
            if el.attrib == xml_tag.dict:
                hits.append(el)
        if len(hits) == 0:
            raise ValueError('No matching attribute')

        if order == - 1:
            hits[0].text = self.join(text_list)
        else:
            hits[order].text = self.join(text_list)


    def replace_xml_tag(self, xml_tag, value, position = -1, order = -1, split = True):
        '''It is assumed that there are one or more instances of this tag. Tags with different dictionary values, are considered to be different. For example,
        <Variable Name="zopa">42</Variable> will only lead to replacement of Variable tags with the name "zopa", but no others.
        No instances results in a no-op. This function replaces the text value of a text by
        a desired value. This value can be numerical or a string. If there is only one version of the text, order parameter need
        not be set. If there are more you must use the order parameter to indicate which tag - in order of appearance - should be modified.
        By default the tag text is split by white space. If  such a split results in more than one item, you must use the position
        parameter to indicate which item should be changed. Any more complex operations are better done directly on the XML tree itself.'''

        text_list = self.get_text_attribute(xml_tag, split)
        if len(text_list) == 0:
            return


        if len(text_list) > 1 and order == -1:
            raise ValueError('There are multiple instances of this tag and you need to specify which one to replace with the order parameter')


        for item in text_list:
            if len(item) > 1 and position == -1:
                raise ValueError('You have a text line split into items. A position parameter is required.')

            if position >= len(item):
                raise ValueError('Position parameter required if you require splitting.')

            item[position]=str(value)

            self.insert_text_attribute(xml_tag, item, order)
        return


    def write(self,file_name):
        self.tree.write(file_name)


if __name__ == "__main__":
    print('Not intended to run as a standlone script.')
