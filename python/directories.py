import os
import errno
import string

# global variable to hold absolute path
ABS_PATH=''
# global variable to hold the root of the miind tree, i.e. the absolute
# path name of the directory that hold the 'miind-git' directory
MIIND_ROOT=''
PATH_VARS_DEFINED=False

def initialize_global_variables():
    global ABS_PATH
    ABS_PATH=os.getcwd()
    global MIIND_ROOT
    MIIND_ROOT =  ABS_PATH[0:-6]
    global PATH_VARS_DEFINED
    PATH_VARS_DEFINED=True

def check_and_strip_name(full_path_name):
    '''Expects full path to the xml file'''
    sep = os.path.sep
    name = full_path_name.split(sep)[-1]
    
    if name[-4:] != '.xml':
        raise NameError
    else:
        return name[:-4]

def miind_root():
    global MIIND_ROOT
    initialize_global_variables()
    return MIIND_ROOT

def create_dir(name):
    ''' Name of the executable to be generated. Should not end in '.xml' '''
    sep = os.path.sep
    initialize_global_variables()
    global MIIND_ROOT

    abs_path = MIIND_ROOT + sep + 'apps' + sep + name

    try:
        os.makedirs(abs_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return abs_path

def insert_cmake_template(name,full_path_name):
    ''' name is the executable name, full_path is the directory where the cmake template
    needs to be written into.'''

    sep = os.path.sep
    outname = full_path_name + sep + 'CMakeLists.txt'
    if os.path.exists(outname):
        return

    with open('cmake_template') as f:
        lines=f.readlines()
    with open(outname,'w') as fout:
        for line in lines:
            fout.write(line)

        fout.write('\nadd_executable( ' + name + ' ' + name + '.cpp)\n')
        fout.write('target_link_libraries( ' + name  + ' ${LIBLIST} )\n')

def parse_apps_cmake(file):
    lines = file.readlines()

    return lines

def generate_apps_cmake(l, dir_path):
    path=os.path.split(os.path.normpath(dir_path))[0]
    cpath= os.path.join(path,'CMakeLists.txt')
    
    with open(cpath,'w') as f:
        for line in l:
            f.write(line)

def executable_list(l):
    ret = []
    '''the executable name is preceded by ADD_SUBDIRECTORY and between brackets'''
    for name in l:
        exec_name = name[name.find('(') + 1 : name.find(')')].strip()
        ret.append(exec_name)
    return ret

def insert_parent_cmake(prog_name, dir_path):
    ''' add the prog_name to the executables in CMakeLists.txt' in the apps directory.'''
    path=os.path.split(os.path.normpath(dir_path))[0]

    fn= os.path.join(path,'CMakeLists.txt')

    with open(fn) as f:
        l=parse_apps_cmake(f)
    if not prog_name in executable_list(l):
        l.append('ADD_SUBDIRECTORY( ' + prog_name + ' )\n')
        generate_apps_cmake(l, dir_path)
    
def add_executable(name,versions=None):
    ''' Add a user defined executable to miind's compilation tree.

    If only a name is provided, but no versions argument, the name
    is expected to be an xml file, with an xml extension. The file name
    with the extension stripped is used to create a directory in the 'apps' subdirectory
    in the miind code tree. CMake files will be added in the appropriate directory.
    If miind was compiled successfully previously, typing 'make' in the 'build'
     subdirectory will cause the new executable to part of the build structure.

    If the versions list is not empty, the name of the xml file with the extension
    stripped will be created in the 'apps' subdirectory. Then for each element in the version
    list a subdirectory will be created, and each of these subdirectories will correspond to an executable,
    that will be part of the build sub structure.
     '''

    global PATH_VARS_DEFINED    
    if not PATH_VARS_DEFINED:
        initialize_global_variables()

    prog_name = check_and_strip_name(name)
    dir_path = create_dir(prog_name)   

    if versions != None:
        for version in versions:
            versname = prog_name + '/' + version
            insert_cmake_template(versname,dir_path)
    else:
        insert_parent_cmake(prog_name, dir_path)
        insert_cmake_template(prog_name,dir_path)        

if __name__ == "__main__":
    initialize_global_variables()
    add_executable('omurtag.xml')
