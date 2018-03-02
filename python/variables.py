import algorithms

def parse_variables(variable_list,outfile):
    s = ''
    for variable in variable_list:
        name  = variable.attrib['Name']
        value = variable.text

        val_type = None
        if 'Type' in variable.attrib :
            val_type = variable.attrib['Type']
        # Allow the user to define a C++-style data type
        if val_type != None :
            s += 'const ' + val_type + ' ' + name + ' = ' + value + ';\n'
        # Otherwise, attempt to auto-detect the type here
        # Even if a number doesn't have a decimal point, assume it's a double
        elif value.replace('.','',1).isdigit() :
            s += 'const double ' + name + ' = ' + value +';\n'
        # Otherwise, assume it's a string
        else :
            s += 'const std::string ' + name + ' = \"' + value +'\";\n'
    s += '\n'
    outfile.write(s)
