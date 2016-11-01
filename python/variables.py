import algorithms

def parse_variables(variable_list,outfile):
    s = ''
    for variable in variable_list:
        name  = variable.attrib['Name']
        value = variable.text
        s += '\tconst double ' + name + ' = ' + value +';\n'
        outfile.write(s)
