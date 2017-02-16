import os
import errno
import string
import inspect
import codegen
import subprocess as sp

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
    ''' Name of the executable to be generated. Should not end in '.xml'. The directory will be created relative to the calling directory. '''
    sep = os.path.sep
    initialize_global_variables()
    global MIIND_ROOT

    abs_path = './' + sep + name

    try:
        os.makedirs(abs_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return abs_path

def filter(lines):
    nw = []
    for line in lines:
        if '${CMAKE_SOURCE_DIR}' in line:
            hh=string.replace(line,'${CMAKE_SOURCE_DIR}',MIIND_ROOT)
            nw.append(hh)
        else:
            nw.append(line)
    return nw

def insert_cmake_template(name,full_path_name):
    ''' name is the executable name, full_path is the directory where the cmake template
    needs to be written into.'''

    sep = os.path.sep
    outname = full_path_name + sep + 'CMakeLists.txt'
    if os.path.exists(outname):
        return

    with open(miind_root() + sep + 'python' + sep + 'cmake_template') as f:
        lines=f.readlines()

    # filter the template CMakeLists.txt to that was is needed locally
    replace = filter(lines)

    with open(outname,'w') as fout:
        for line in replace:
            fout.write(line)

        # add  the miind libraries explicitly
        libbase = MIIND_ROOT + '/build/libs'
        numdir  = libbase + '/NumtoolsLib'
        geomdir = libbase + '/GeomLib'
        mpidir  = libbase + '/MPILib'
        twodir  = libbase + '/TwoDLib'
        fout.write('link_directories(' + numdir + ' ' + geomdir + ' ' + mpidir + ' ' + twodir + ')\n')
        fout.write('\nadd_executable( ' + name + ' ' + name + '.cpp)\n')
        fout.write('target_link_libraries( ' + name  + ' ${LIBLIST} )\n')


def create_cpp_file(name, dir_path, prog_name, mod_name):

    cpp_name = prog_name + '.cpp'
    abs_path = os.path.join(dir_path,cpp_name)
    with open(abs_path,'w') as fout:
        with open(name) as fin:
            codegen.generate_outputfile(fin,fout)

    if mod_name != None:
        for f in mod_name:
            sp.call(['cp',f,dir_path])
    return
            
    
def add_executable(dirname, xmlfiles, modname):

    ''' Add a user defined executable to the current working directory.
     '''
    global PATH_VARS_DEFINED    
    if not PATH_VARS_DEFINED:
        initialize_global_variables()

    sep = os.path.sep
    if len(xmlfiles) == 1:
        dirpath = create_dir(dirname)
        progname = check_and_strip_name(xmlfiles[0])
        insert_cmake_template(progname,dirpath)
        create_cpp_file(xmlfiles[0], dirpath, progname, modname)
    else:
        for xmlfile in xmlfiles:
            progname = check_and_strip_name(xmlfile)
            dirpath = create_dir(dirname + '/' + progname)
            insert_cmake_template(progname,dirpath)
            create_cpp_file(xmlfile, dirpath, progname, modname)

if __name__ == "__main__":
    initialize_global_variables()
