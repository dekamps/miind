import os
import errno
import string
import inspect
import codegen

# global variable to hold absolute path
ABS_PATH=''
# global variable to hold the root of the miind tree, i.e. the absolute
# path name of the directory that hold the 'miind-git' directory
MIIND_ROOT=''
PATH_VARS_DEFINED=False

def initialize_global_variables():
    global ABS_PATH

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))    
    ABS_PATH=path
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
    ''' Name of the executable to be generated. Should not end in '.xml'. The directory will be created relative to MIIND_ROOT. '''
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

    with open(miind_root() + sep + 'python' + sep + 'cmake_template') as f:
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
    ''' First, the last argument is stripped from dir_path. If a CMakeLists.txt file is not present in the resulting directory, one will be created. If there is one,
    a line will be added for each executable present in l, except if it is already present in the CMakeLists.txt'''
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

    if not os.path.exists(fn):
        with open(fn,'w') as f:
            f.write('# Machine generated CMakeLists.txt file by directories.py\n')
            
    with open(fn) as f:
        l=parse_apps_cmake(f)
    if not prog_name in executable_list(l):
        l.append('ADD_SUBDIRECTORY( ' + prog_name + ' )\n')
        generate_apps_cmake(l, dir_path)

def detach_executable(name):
    ''' Remove an executable or execuatble tree from the build tree. After calling a 'make' command
    will no longer trry to compile the executable 'name', or the executables under the directory 'name'.
    No files are removed.'''

    dir_path =path=os.path.join(miind_root(),'apps')
    
    fn = os.path.join(dir_path,'CMakeLists.txt')
    
    with open(fn) as f:
        l=parse_apps_cmake(f)

    if name in executable_list(l):
        index = executable_list(l).index(name)
        del l[index]
        # generate_apps_cmake operates on parent directory, therefore the dummy argument
        dir_path = os.path.join(dir_path,'dummy')
        generate_apps_cmake(l, dir_path)
    else:
        print 'Name not in excutable list. No action taken.'

def create_cpp_file(name, dir_path, prog_name):

    cpp_name = prog_name + '.cpp'
    abs_path = os.path.join(dir_path,cpp_name)
    with open(abs_path,'w') as fout:
        with open(name) as fin:
            codegen.generate_outputfile(fin,fout)
    return
            
    
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

    sep = os.path.sep

    if versions != None:
        dir_path = create_dir(name)
        insert_parent_cmake(name,dir_path)
        for version in versions:
            prog_name = check_and_strip_name(version)
            versname = name + sep + prog_name
            dir_path = create_dir(versname)

            insert_parent_cmake(prog_name, dir_path)
            insert_cmake_template(prog_name,dir_path)
            create_cpp_file(version, dir_path, prog_name)
    else:
        prog_name = check_and_strip_name(name)
        dir_path = create_dir(prog_name)
    
        insert_parent_cmake(prog_name, dir_path)
        insert_cmake_template(prog_name,dir_path)
        create_cpp_file(name, dir_path, prog_name)

if __name__ == "__main__":
    initialize_global_variables()
    add_executable('masterblaster', ['omurtag.xml', 'twopop.xml'])
    detach_executable('masterblaster')
