import os
import errno
import string
import inspect
import shutil
import miind.codegen3 as codegen
import miind.codegen_lib3 as codegen_lib
from shutil import copyfile, copy2

# global variable to hold absolute path
ABS_PATH=''

PATH_VARS_DEFINED=False


def initialize_global_variables():
    '''Determines the absolute path name to the directory where the script is aclled from. '''
    global ABS_PATH
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    ABS_PATH=path
    global PATH_VARS_DEFINED
    PATH_VARS_DEFINED=True

def check_and_strip_name(full_path_name):
    '''Expects full path to the xml file.'''
    sep = os.path.sep
    name = full_path_name.split(sep)[-1]

    if name[-4:] != '.xml':
        raise NameError
    else:
        return name[:-4]

def miind_python_dir():
    global ABS_PATH
    initialize_global_variables()
    return ABS_PATH

def create_dir(name):
    abs_path=os.path.join('.',name)
    try:
        os.makedirs(abs_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return abs_path

def filter(lines,enable_mpi,enable_openmp,enable_root, source_file_name):
    # ROOT enabled by default
    if enable_root:
        ENABLE_ROOT = 'ON'
    else:
        ENABLE_ROOT = 'OFF'

    # MPI disabled by default
    if enable_mpi:
        ENABLE_MPI = 'ON'
    else:
        ENABLE_MPI = 'OFF'

    # OPENMP disabled by default
    if enable_openmp:
        ENABLE_OPENMP = 'ON'
    else:
        ENABLE_OPENMP = 'OFF'

    nw = []
    for line in lines:
        if '${CMAKE_SOURCE_DIR}' in line:
            p = os.path.join(miind_python_dir(), '..','..')
            miind_root = p.replace('\\', '/')
            hh=line.replace('${CMAKE_SOURCE_DIR}',miind_root)
            nw.append(hh)
        elif '${TOKEN_ENABLE_MPI}' in line:
    	    hh=line.replace('${TOKEN_ENABLE_MPI}',ENABLE_MPI)
    	    nw.append(hh)
        elif '${TOKEN_ENABLE_OPENMP}' in line:
    	    hh=line.replace('${TOKEN_ENABLE_OPENMP}',ENABLE_OPENMP)
    	    nw.append(hh)
        elif '${TOKEN_ENABLE_ROOT}' in line:
    	    hh=line.replace('${TOKEN_ENABLE_ROOT}',ENABLE_ROOT)
    	    nw.append(hh)
        elif 'TOKEN_SOURCE_FILE' in line:
    	    hh=line.replace('TOKEN_SOURCE_FILE',source_file_name)
    	    nw.append(hh)
        else:
            nw.append(line)
    return nw

def insert_cmake_template_lib(name, full_path_name, enable_mpi,enable_openmp,enable_root,cuda,source_file_name):
    ''' name is the executable name, full_path is the directory where the cmake template
    needs to be written into.'''

    # If we're rebuilding CMakeLists.txt, then we need to clear CMakeCache to avoid unexpected
    # behaviour.
    cachefile = os.path.join(full_path_name, 'CMakeCache.txt')
    if os.path.exists(cachefile):
	       os.remove(cachefile)

    outname = os.path.join(full_path_name, 'CMakeLists.txt')
    template_path = os.path.join(miind_python_dir(),'cmake_template_lib')
    with open(template_path) as f:
        lines=f.readlines()

    # filter the template CMakeLists.txt to that was is needed locally
    replace = filter(lines,enable_mpi,enable_openmp,enable_root,source_file_name)

    with open(outname,'w') as fout:
        if cuda == True:
            fout.write('OPTION(ENABLE_CUDA \"CUDA Desired\" ON)\n')
        else:
            fout.write('OPTION(ENABLE_CUDA \"CUDA Desired\" OFF)\n')

        for line in replace:
            fout.write(line)

        libbase = os.path.join(miind_python_dir(), '..', '..', 'build','libs')
        numdir  = os.path.join(libbase,'NumtoolsLib')
        geomdir = os.path.join(libbase,'GeomLib')
        mpidir  = os.path.join(libbase,'MPILib')
        twodir  = os.path.join(libbase,'TwoDLib')


        if cuda == True:
            cudatwodir = os.path.join(libbase,'CudaTwoDLib')
            shared  = os.path.join(libbase,'MiindLib')
        else:
            cudatwodir = ''
            shared  = ''

        # If MIIND_INCLUDE_DIR has been defined, we're compiling for a development version (not installed version)
        # so set link to the development libraries. Otherwise, libraries are in usr/lib where they should be.
        fout.write('IF (MIIND_INCLUDE_DIR)\n')
        if(os.path.isfile(libbase)): # maybe we're in python...
            fout.write('link_directories(' + numdir + ' ' + geomdir + ' ' + mpidir + ' ' + twodir + ' ' + cudatwodir + ' ' + shared +')\n')
        else:
            libbase = os.path.join(miind_python_dir(), 'build','libs')
            fout.write('link_directories('+ libbase +')\n')
        fout.write('ENDIF (MIIND_INCLUDE_DIR)\n')
        if cuda == True:
            fout.write('\ncuda_add_library( '+ name + ' ${LIB_TYPE} ${TVB_LIF_SRC} ${PW_HEADERS})\n')
        else:
            fout.write('\nadd_library( '+ name + ' ${LIB_TYPE} ${TVB_LIF_SRC} ${PW_HEADERS})\n')

        fout.write('target_link_libraries( ' + name + ' ${LIBLIST})\n')



def insert_cmake_template(name,full_path_name,enable_mpi,enable_openmp,enable_root,cuda,source_file_name):
    ''' name is the executable name, full_path is the directory where the cmake template
    needs to be written into.'''

    outname = os.path.join(full_path_name, 'CMakeLists.txt')
    if os.path.exists(outname):
        return
    template_path = os.path.join(miind_python_dir(),'cmake_template')
    with open(template_path) as f:
        lines=f.readlines()

    # filter the template CMakeLists.txt to that was is needed locally
    replace = filter(lines,enable_mpi,enable_openmp,enable_root,source_file_name)

    with open(outname,'w') as fout:
        if cuda == True:
            fout.write('OPTION(ENABLE_CUDA \"CUDA Desired\" ON)\n')
        else:
            fout.write('OPTION(ENABLE_CUDA \"CUDA Desired\" OFF)\n')

        for line in replace:
            fout.write(line)


        # add  the miind libraries explicitly
        
        libbase = os.path.join(miind_python_dir(), '..', '..', 'build','libs')
        numdir  = os.path.join(libbase,'NumtoolsLib')
        geomdir = os.path.join(libbase,'GeomLib')
        mpidir  = os.path.join(libbase,'MPILib')
        twodir  = os.path.join(libbase,'TwoDLib')

        if cuda == True:
            cudatwodir = os.path.join(libbase,'CudaTwoDLib')
            shared  = os.path.join(libbase,'MiindLib')
        else:
            cudatwodir = ''
            shared  = ''

        # If MIIND_INCLUDE_DIR has been defined, we're compiling for a development version (not installed version)
        # so set link to the development libraries. Otherwise, libraries are in usr/lib where they should be.
        fout.write('IF (MIIND_INCLUDE_DIR)\n')
        if(os.path.isfile(libbase)): # maybe we're in python...
            fout.write('link_directories(' + numdir + ' ' + geomdir + ' ' + mpidir + ' ' + twodir + ' ' + cudatwodir + ' ' + shared +')\n')
        else:
            libbase = os.path.join(miind_python_dir(), 'build','libs')
            fout.write('link_directories('+ libbase +')\n')
        fout.write('ENDIF (MIIND_INCLUDE_DIR)\n')

        if cuda == True:
            fout.write('\ncuda_add_executable( ' + name + ' ' + name + '.cu)\n')
        else:
            fout.write('\nadd_executable( ' + name + ' ' + name + '.cpp)\n')
        fout.write('target_link_libraries( ' + name  + ' ${LIBLIST} )\n')

def create_cpp_lib_file(name, dir_path, prog_name, mod_name, enable_root):

    cpp_name = prog_name + '.cpp'
    abs_path = os.path.join(dir_path,cpp_name)
    with open(abs_path,'w') as fout:
        with open(name) as fin:
            codegen_lib.generate_outputfile(fin,fout,prog_name)

    if mod_name != None:
        for f in mod_name:
            copyfile(f,dir_path)
    return

def create_cpp_file(name, dir_path, prog_name, mod_name, enable_root):

    cpp_name = prog_name + '.cpp'
    abs_path = os.path.join(dir_path,cpp_name)
    with open(abs_path,'w') as fout:
        with open(name) as fin:
            codegen.generate_outputfile(fin,fout, enable_root)

    if mod_name != None:
        for f in mod_name:
            shutil.copy2(f,dir_path)
    return


def move_model_files(xmlfile,dirpath):
    '''Collect the model files, and the matrix files that are mentioned in the XML file,
    and move them to directory dirpath.'''
    mns =  codegen.model_name(xmlfile)
    mans = codegen.matrix_names(xmlfile)
    tms = codegen.matrix_transform_name(xmlfile)

    for model in mns:
        if not os.path.exists(model):
            print('Please put the file: ', model, 'in the same directory as the xml file.')

    for tmat in tms:
        if not os.path.exists(tmat):
            print('Please put the file: ', tmat, 'in the same directory as the xml file.')

    for mat in mans:
        if not os.path.exists(mat):
            print('Please put the file: ', mat, 'in the same directory as the xml file.')

    fls = mans + mns + tms
    for fi in fls:
        shutil.copy2(fi,dirpath)

def add_shared_library(dirname, xmlfiles, modname, enable_mpi=True, enable_openmp=True, enable_root=True, enable_cuda=False):

    ''' Add a user defined executable to the current working directory.
     '''
    global PATH_VARS_DEFINED
    if not PATH_VARS_DEFINED:
        initialize_global_variables()

    for xmlfile in xmlfiles:
        progname = check_and_strip_name(xmlfile)
        dirpath = create_dir(os.path.join(dirname, progname))
        SOURCE_FILE = progname + '.cpp'
        insert_cmake_template_lib(progname,dirpath,enable_mpi,enable_openmp,enable_root,enable_cuda,SOURCE_FILE)
        create_cpp_lib_file(xmlfile, dirpath, progname, modname, enable_root)
        move_model_files(xmlfile,dirpath)
        xmlfilename = xmlfile.split(os.path.sep)[-1]
        shutil.copy2(xmlfile, os.path.join(dirpath,xmlfilename))

def add_executable(dirname, xmlfiles, modname,enable_mpi=True, enable_openmp=True, enable_root=True, enable_cuda=False):
    ''' Add a user defined executable to the current working directory.
     '''
    global PATH_VARS_DEFINED
    if not PATH_VARS_DEFINED:
        initialize_global_variables()

    for xmlfile in xmlfiles:
        progname = check_and_strip_name(xmlfile)
        dirpath = create_dir(os.path.join(dirname, progname))
        SOURCE_FILE = progname + '.cpp'
        insert_cmake_template(progname,dirpath,enable_mpi,enable_openmp,enable_root,enable_cuda,SOURCE_FILE)
        create_cpp_file(xmlfile, dirpath, progname, modname, enable_root)
        move_model_files(xmlfile,dirpath)
        xmlfilename = xmlfile.split(os.path.sep)[-1]
        shutil.copyfile(xmlfile, os.path.join(dirpath,xmlfilename))

if __name__ == "__main__":
    initialize_global_variables()
