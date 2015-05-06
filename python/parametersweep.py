import string
import xml.etree.ElementTree as ET

class xml_tag_convertor:
    '''Take an xml tag directly frpm a simulation XML file.
    Convert it ino a tag/dictonary tuple'''

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
        self.name = file_name
        self.tree = ET.parse(file_name)

    def get_text_attribute(self, xml_tag):
        root = self.tree.getroot()
        l=root.iter(xml_tag.name)
        count = 0
        for el in l:
            if el.attrib == xml_tag.dict:
                store = el
                count += 1

        if count != 1:
            raise
        textlist = el.text.split()
        return textlist

    def join(self,l):
        s = ''
        for el in l[:-1]:
            s += el
            s += ' '
        s += l[-1]

        return s

    def insert_text_attribute(self, xml_tag, text_list):        
        root = self.tree.getroot()
        l=root.iter(xml_tag.name)
        for el in l:
            if el.attrib == xml_tag.dict:
                el.text = self.join(text_list)


    def replace_xml_tag(self, xml_tag, value, position = 0):

        text_list = self.get_text_attribute(xml_tag)

        '''it is assumed that the value is a numeral, i.e. it will need
        to be converted to a string in the tag list.'''
    
        if position >= len(text_list):
            raise
        text_list[position]=str(value)      

        self.insert_text_attribute(xml_tag,text_list)
        return


    def write(self,file_name):
        self.tree.write(file_name)
 

if __name__ == "__main__":

    conv = xml_tag_convertor('<Connection In="Cortical Background" Out="LIF E">800 0.03 0</Connection>')

    tag=xml_tag(conv.convert())
    file = xml_file('xml/omurtag.xml')

    file.replace_xml_tag(tag,500.0,0)
    file.replace_xml_tag(tag,0.1,1)
#    file.write('kadaver.xml')
    
       
