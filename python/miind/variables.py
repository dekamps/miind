import miind.algorithms as algorithms

VARIABLES = []

def parse_variables(variable_list,outfile):
    global VARIABLES
    s = ''
    for variable in variable_list:
        name  = variable.attrib['Name']
        value = variable.text
        if name not in VARIABLES:
            VARIABLES = VARIABLES + [name]

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

def parse_variables_as_parameters(variable_list, outfile):
    global VARIABLES
    s = ''
    for variable in variable_list:
        name  = variable.attrib['Name']
        value = variable.text
        if name not in VARIABLES:
            VARIABLES = VARIABLES + [name]

        val_type = None
        if 'Type' in variable.attrib :
            val_type = variable.attrib['Type']

        if val_type != None :
            s += '\n\t\t\t, const ' + val_type + ' _' + name
        # Otherwise, attempt to auto-detect the type here
        # Even if a number doesn't have a decimal point, assume it's a double
        elif value.replace('.','',1).isdigit() :
            s += '\n\t\t\t, const double _' + name
        # Otherwise, assume it's a string
        else :
            s += '\n\t\t\t, const std::string _' + name
    outfile.write(s[5:])

def parse_variables_as_constructor_defaults(variable_list, outfile):
    global VARIABLES
    s = ''
    for variable in variable_list:
        name  = variable.attrib['Name']
        if name not in VARIABLES:
            VARIABLES = VARIABLES + [name]
        s += '\n\t\t\t,' + name + '(_' + name + ')'
    outfile.write(s)

def parse_variable_types(variable_list):
    s = ''
    for variable in variable_list:
        value = variable.text
        val_type = None
        if 'Type' in variable.attrib :
            val_type = variable.attrib['Type']

        if val_type != None :
            s += ',const ' + val_type
        # Otherwise, attempt to auto-detect the type here
        # Even if a number doesn't have a decimal point, assume it's a double
        elif value.replace('.','',1).isdigit() :
            s += ',const double'
        # Otherwise, assume it's a string
        else :
            s += ',const std::string'

    return s

def parse_variable_python_def(variable_list):
    s = ''
    fmt = 'i'
    for variable in variable_list:
        name  = variable.attrib['Name']
        value = variable.text

        var_type = variable.attrib['Type']

        if var_type == 'int':
            fmt += 'i'
            s += '\tint ' + name + ';\n'
        ### Add further types here as they come up...
        elif value.replace('.','',1).isdigit() :
            fmt += 'd'
            s += '\tdouble ' + name + ';\n'
        # Otherwise, assume it's a string
        else :
            fmt = fmt + 's'
            s += '\tstd::string ' + name + ';\n'

    write = ''
    write += s
    write += '\tint nodes;'
    write += '\n'
    write += '\tif (!PyArg_ParseTuple(args, \"' + fmt + '\", &nodes'
    for variable in variable_list:
        name  = variable.attrib['Name']
        write += ',&'+ name
    write += '\t))\n'
    write += '\t\tmodel = new MiindModel();\n'
    write += '\telse\n'
    write += '\t\tmodel = new MiindModel(nodes'
    for variable in variable_list:
        name  = variable.attrib['Name']
        write += ',' + name
    write += ');\n'
    write += '\tmodel->init();\n'
    return write

def variable_or_string(s):
    global VARIABLES
    if s in VARIABLES:
        return s
    return "\"" + s + "\""