import os
import errno
import string
import inspect
import miind.codegen as codegen
import miind.codegen_lib as codegen_lib
import subprocess as sp

# global variable to hold absolute path
ABS_PATH=''
# global variable to hold the root of the miind tree, i.e. the absolute
# path name of the directory that hold the 'miind-git' directory
MIIND_ROOT=''
PATH_VARS_DEFINED=False
# global variable to hold ENABLE_MPI (ON/OFF)
ENABLE_MPI='OFF'
# global variable to hold ENABLE_OPENMP (ON/OFF)
ENABLE_OPENMP='OFF'
# global variable to hold ENABLE_ROOT (ON/OFF)
ENABLE_ROOT='ON'
# global variable to hold ENABLE_ROOT (ON/OFF)
ENABLE_CUDA='ON'
# global variable to hold CPP source file (for use with building the shared library)
SOURCE_FILE='MiindModel.cpp'

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
    initialize_global_variables()
    global MIIND_ROOT

    abs_path=os.path.join('.',name)
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
    	elif '${TOKEN_ENABLE_MPI}' in line:
    	    hh=string.replace(line,'${TOKEN_ENABLE_MPI}',ENABLE_MPI)
    	    nw.append(hh)
    	elif '${TOKEN_ENABLE_OPENMP}' in line:
    	    hh=string.replace(line,'${TOKEN_ENABLE_OPENMP}',ENABLE_OPENMP)
    	    nw.append(hh)
    	elif '${TOKEN_ENABLE_ROOT}' in line:
    	    hh=string.replace(line,'${TOKEN_ENABLE_ROOT}',ENABLE_ROOT)
    	    nw.append(hh)
        elif 'TOKEN_SOURCE_FILE' in line:
    	    hh=string.replace(line,'TOKEN_SOURCE_FILE',SOURCE_FILE)
    	    nw.append(hh)
        else:
            nw.append(line)
    return nw

def insert_cmake_template_lib(name, full_path_name):
    ''' name is the executable name, full_path is the directory where the cmake template
    needs to be written into.'''

    # If we're rebuilding CMakeLists.txt, then we need to clear CMakeCache to avoid unexpected
    # behaviour.
    cachefile = os.path.join(full_path_name, 'CMakeCache.txt')
    if os.path.exists(cachefile):
	       os.remove(cachefile)

    outname = os.path.join(full_path_name, 'CMakeLists.txt')
    template_path = os.path.join(miind_root(),'python','cmake_template_lib')
    with open(template_path) as f:
        lines=f.readlines()

    # filter the template CMakeLists.txt to that was is needed locally
    replace = filter(lines)

    with open(outname,'w') as fout:
        for line in replace:
            fout.write(line)

        libbase = MIIND_ROOT + '/build/libs'
        numdir  = libbase + '/NumtoolsLib'
        geomdir = libbase + '/GeomLib'
        mpidir  = libbase + '/MPILib'
        twodir  = libbase + '/TwoDLib'
        fout.write('link_directories(' + numdir + ' ' + geomdir + ' ' + mpidir + ' ' + twodir + ')\n')
        fout.write('\nadd_library( '+ name + ' ${LIB_TYPE} ${TVB_LIF_SRC} ${PW_HEADERS})\n')
        fout.write('target_link_libraries( ' + name + ' ${LIBLIST} -lpython2.7 -lboost_python)\n')


def insert_cmake_template(name,full_path_name):
    ''' name is the executable name, full_path is the directory where the cmake template
    needs to be written into.'''

    # If we're rebuilding CMakeLists.txt, then we need to clear CMakeCache to avoid unexpected
    # behaviour.
    cachefile = os.path.join(full_path_name, 'CMakeCache.txt')
    if os.path.exists(cachefile):
	os.remove(cachefile)

    outname = os.path.join(full_path_name, 'CMakeLists.txt')
    template_path = os.path.join(miind_root(),'python','cmake_template')
    with open(template_path) as f:
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


def create_cpp_lib_file(name, dir_path, prog_name, mod_name):

    cpp_name = prog_name + '.cpp'
    abs_path = os.path.join(dir_path,cpp_name)
    with open(abs_path,'w') as fout:
        with open(name) as fin:
            codegen_lib.generate_outputfile(fin,fout)

    if mod_name != None:
        for f in mod_name:
            sp.call(['cp',f,dir_path])
    return

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


def move_model_files(xmlfile,dirpath):
    '''Collect the model files, and the matrix files that are mentioned in the XML file,
    and move them to directory dirpath.'''
    mns =  codegen.model_name(xmlfile)
    mans = codegen.matrix_names(xmlfile)
    tms = codegen.matrix_transform_name(xmlfile)

    for model in mns:
        if not os.path.exists(model):
            print 'Please put the file: ', model, 'in the same directory as the xml file.'

    for tmat in tms:
        if not os.path.exists(tmat):
            print 'Please put the file: ', tmat, 'in the same directory as the xml file.'

    for mat in mans:
        if not os.path.exists(mat):
            print 'Please put the file: ', mat, 'in the same directory as the xml file.'

    fls = mans + mns + tms
    for fi in fls:
        sp.call(['cp',fi,dirpath])

def add_shared_library(dirname, xmlfiles, modname, enable_mpi=False, enable_openmp=False, enable_root=True):
    global ENABLE_MPI
    global ENABLE_OPENMP
    global ENABLE_ROOT
    global SOURCE_FILE

    ''' Add a user defined executable to the current working directory.
     '''
    global PATH_VARS_DEFINED
    if not PATH_VARS_DEFINED:
        initialize_global_variables()

    # ROOT enabled by default
    if not enable_root:
	ENABLE_ROOT = 'OFF'

    # MPI disabled by default
    if enable_mpi:
	ENABLE_MPI = 'ON'

    # OPENMP disabled by default
    if enable_openmp:
	ENABLE_OPENMP = 'ON'

    for xmlfile in xmlfiles:
        progname = check_and_strip_name(xmlfile)
        dirpath = create_dir(os.path.join(dirname, progname))
        SOURCE_FILE = progname + '.cpp'
        insert_cmake_template_lib(progname,dirpath)
        create_cpp_lib_file(xmlfile, dirpath, progname, modname)
        move_model_files(xmlfile,dirpath)

def add_executable(dirname, xmlfiles, modname, enable_mpi=False, enable_openmp=False, enable_root=True, enable_cuda=False):
    global ENABLE_MPI
    global ENABLE_OPENMP
    global ENABLE_ROOT
    global ENABLE_CUDA
    global SOURCE_FILE

    ''' Add a user defined executable to the current working directory.
     '''
    global PATH_VARS_DEFINED
    if not PATH_VARS_DEFINED:
        initialize_global_variables()

    # ROOT enabled by default
    if not enable_root:
        ENABLE_ROOT = 'OFF'

    # MPI disabled by default
    if enable_mpi:
        ENABLE_MPI = 'ON'

    # OPENMP disabled by default
    if enable_openmp:
        ENABLE_OPENMP = 'ON'

    # CUDA disabled by default
    if enable_cuda:
        ENABLE_CUDA = 'ON'

    for xmlfile in xmlfiles:
        progname = check_and_strip_name(xmlfile)
        dirpath = create_dir(os.path.join(dirname, progname))
        SOURCE_FILE = progname + '.cpp'
        insert_cmake_template(progname,dirpath)
        create_cpp_file(xmlfile, dirpath, progname, modname)
        move_model_files(xmlfile,dirpath)

if __name__ == "__main__":
    initialize_global_variables()
