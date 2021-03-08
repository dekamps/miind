import io
import os
import os.path
import pathlib
import sys
import runpy
import subprocess
import re
import sysconfig
import shutil
import platform
from sys import platform
import skbuild
from skbuild import cmaker

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cmake_source_dir = "."
    
    # Build empty miind and miind_api directories in root folder. 
    # We don't want them there in the repo as they're just placeholders.
    miind_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'miind')
    try:
        os.mkdir(miind_dir)
    except:
        print("Could not create miind directory", miind_dir)
    miind_api_dir = os.path.join(miind_dir, 'miind_api')
    try:
        os.mkdir(miind_api_dir)
    except:
        print("Could not create miind_api directory", miind_api_dir)

    python_version = cmaker.CMaker.get_python_version()
    python_lib_path = cmaker.CMaker.get_python_library(python_version).replace(
        "\\", "/"
    )
    python_include_dir = cmaker.CMaker.get_python_include_dir(python_version).replace(
        "\\", "/"
    )

    package_version = "0.0.11"

    package_name = "miind"

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    packages = ["miind", "miind.miind_api"]

    package_data = {
        "miind": [],
        "miind.testfiles": [],
        "miind.miind_api": [],
        "miind.build": [],
        "miind.build.apps.MatrixGenerator": [],
        "miind.build.apps.Projection": [],
        "miind.build.examples": []
    }

    # Files from CMake output to copy to package.
    # Path regexes with forward slashes relative to CMake install dir.
    rearrange_cmake_output_data = {
        "miind": (["lib/miindsim.pyd","lib/miindsimv.pyd"] if platform == "win32" else []) + (["bin\/.+\.dll"] if platform == "win32" else []) + (["lib/miindsim.so"] if platform not in ["win32"] else []) + (["lib/miindsimv.so"] if platform not in ["win32","darwin"] else []) + ["share\/miind\/python\/miind\/.+"],
        "miind.miind_api": ["share\/miind\/python\/miind\/miind_api\/.+"],
        "miind.build": [],
        "miind.build.apps.MatrixGenerator": (["bin\/gsl.dll", "bin\/gslcblas.dll", "bin\/vcruntime140.dll", "bin\/vcruntime140_1.dll", "bin\/msvcp140.dll"] if platform == "win32" else []) + (["bin\/MatrixGenerator.+", "bin\/Bind.+"] if platform == "win32" else []) + (["share\/miind\/apps\/MatrixGenerator\/.+"] if platform != "win32" else []),
        "miind.build.apps.Projection": (["bin\/gsl.dll", "bin\/gslcblas.dll", "bin\/vcruntime140.dll", "bin\/vcruntime140_1.dll", "bin\/msvcp140.dll"] if platform == "win32" else []) + (["bin\/Projection.+"] if platform == "win32" else []) + (["share\/miind\/apps\/Projection\/.+"] if platform != "win32" else []),
        "miind.build.examples": ["share\/miind\/examples\/.+"],
        "miind.testfiles": ["share\/miind\/python\/miind\/testfiles\/.+"]
    }

    # Files in sourcetree outside package dir that should be copied to package.
    # Raw paths relative to sourcetree root.
    files_outside_package_dir = {"miind": []}
    
    # Copy triplet files to vcpkg from custom-triplets to vcpkg
    src_linux = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom-triplets/x64-linux-mixed.cmake')
    src_osx = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom-triplets/x64-osx-mixed.cmake')
    #src_windows = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'custom-triplets/x64-windows-mixed.cmake')
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vcpkg/triplets/')
    shutil.copy2(src_linux, dst)
    shutil.copy2(src_osx, dst)
    #shutil.copy2(src_windows, dst)
    
    cmake_args = (
        [
            '-DCMAKE_BUILD_TYPE=Release',
            '-DENABLE_OPENMP:BOOL=ON',
            '-DENABLE_MPI:BOOL=OFF',
            '-DENABLE_TESTING:BOOL=ON',
            '-DENABLE_CUDA:BOOL=ON',
            '-DENABLE_ROOT:BOOL=OFF',
            '-DCMAKE_CUDA_FLAGS=--generate-code=arch=compute_52,code=[compute_52,sm_52]'#, VCPkg installs libraries which are too new for manylinux2014 so we have to do it the old fashioned way. sucks.
            #'-DVCPKG_MANIFEST_INSTALL:BOOL=ON',
            #'-DVCPKG_MANIFEST_MODE:BOOL=ON',
            #'-DVCPKG_APPLOCAL_DEPS:BOOL=ON',
            #'-DVCPKG_TARGET_TRIPLET=x64-linux-mixed',
            #'-DCMAKE_TOOLCHAIN_FILE=' + os.path.dirname(os.path.abspath(__file__)) + '/vcpkg/scripts/buildsystems/vcpkg.cmake'
        ]
    )
        
    if platform == "win32":
        cmake_args = (
            [
                '-DCMAKE_BUILD_TYPE=Release',
                '-DENABLE_OPENMP:BOOL=ON',
                '-DENABLE_MPI:BOOL=OFF',
                '-DENABLE_TESTING:BOOL=ON',
                '-DENABLE_CUDA:BOOL=ON',
                '-DENABLE_ROOT:BOOL=OFF',
                '-DVCPKG_MANIFEST_INSTALL:BOOL=ON',
                '-DVCPKG_MANIFEST_MODE:BOOL=ON',
                '-DVCPKG_APPLOCAL_DEPS:BOOL=ON',
                '-DCMAKE_CUDA_FLAGS=--generate-code=arch=compute_52,code=[compute_52,sm_52]',
                '-DVCPKG_TARGET_TRIPLET=x64-windows',
                '-DCMAKE_TOOLCHAIN_FILE=' + os.path.dirname(os.path.abspath(__file__)) + '/vcpkg/scripts/buildsystems/vcpkg.cmake'
            ]
        )
        
    if platform == "darwin":
        cmake_args = (
            [
                '-DCMAKE_BUILD_TYPE=Release',
                '-DENABLE_OPENMP:BOOL=ON',
                '-DENABLE_MPI:BOOL=OFF',
                '-DENABLE_TESTING:BOOL=ON',
                '-DENABLE_CUDA:BOOL=OFF',
                '-DENABLE_ROOT:BOOL=OFF',
                '-DCMAKE_CUDA_FLAGS=--generate-code=arch=compute_52,code=[compute_52,sm_52]'#,
                #'-DVCPKG_MANIFEST_INSTALL:BOOL=ON',
                #'-DVCPKG_MANIFEST_MODE:BOOL=ON',
                #'-DVCPKG_APPLOCAL_DEPS:BOOL=ON',
                #'-DVCPKG_TARGET_TRIPLET=x64-osx-mixed',
                #'-DCMAKE_TOOLCHAIN_FILE=' + os.path.dirname(os.path.abspath(__file__)) + #'/vcpkg/scripts/buildsystems/vcpkg.cmake'
            ]
        )
    

    # https://github.com/scikit-build/scikit-build/issues/479
    if "CMAKE_ARGS" in os.environ:
        import shlex

        cmake_args.extend(shlex.split(os.environ["CMAKE_ARGS"]))
        del shlex

    # works via side effect
    RearrangeCMakeOutput(
        rearrange_cmake_output_data, files_outside_package_dir, package_data.keys()
    )

    skbuild.setup(
        name=package_name,
        version=package_version,
        url="https://github.com/dekamps/miind",
        license="MIT",
        description="MIIND",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=packages,
        package_data=package_data,
        install_requires=[
              'shapely',
              'descartes',
              'numpy',
              'matplotlib',
              'scipy'
        ],
        python_requires=">=3.6",
        classifiers=[
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: MIT License",
            "Operating System :: MacOS",
            "Operating System :: Unix",
            "Operating System :: Microsoft :: Windows",
        ],
        cmake_args=cmake_args,
        cmake_source_dir=cmake_source_dir,
    )
    


class RearrangeCMakeOutput(object):
    """
        Patch SKBuild logic to only take files related to the Python package
        and construct a file hierarchy that SKBuild expects (see below)
    """

    _setuptools_wrap = None

    # Have to wrap a function reference, or it's converted
    # into an instance method on attr assignment
    import argparse

    wraps = argparse.Namespace(_classify_installed_files=None)
    del argparse

    package_paths_re = None
    packages = None
    files_outside_package = None

    def __init__(self, package_paths_re, files_outside_package, packages):
        cls = self.__class__
        assert not cls.wraps._classify_installed_files, "Singleton object"
        import skbuild.setuptools_wrap

        cls._setuptools_wrap = skbuild.setuptools_wrap
        cls.wraps._classify_installed_files = (
            cls._setuptools_wrap._classify_installed_files
        )
        cls._setuptools_wrap._classify_installed_files = (
            self._classify_installed_files_override
        )

        cls.package_paths_re = package_paths_re
        cls.files_outside_package = files_outside_package
        cls.packages = packages

    def __del__(self):
        cls = self.__class__
        cls._setuptools_wrap._classify_installed_files = (
            cls.wraps._classify_installed_files
        )
        cls.wraps._classify_installed_files = None
        cls._setuptools_wrap = None

    def _classify_installed_files_override(
        self,
        install_paths,
        package_data,
        package_prefixes,
        py_modules,
        new_py_modules,
        scripts,
        new_scripts,
        data_files,
        cmake_source_dir,
        cmake_install_reldir,
    ):
        """
            From all CMake output, we're only interested in a few files
            and must place them into CMake install dir according
            to Python conventions for SKBuild to find them:
                package\
                    file
                    subpackage\
                        etc.
        """

        cls = self.__class__

        # 'relpath'/'reldir' = relative to CMAKE_INSTALL_DIR/cmake_install_dir
        # 'path'/'dir' = relative to sourcetree root
        cmake_install_dir = os.path.join(
            cls._setuptools_wrap.CMAKE_INSTALL_DIR(), cmake_install_reldir
        )
        install_relpaths = [
            os.path.relpath(p, cmake_install_dir) for p in install_paths
        ]
        fslash_install_relpaths = [
            p.replace(os.path.sep, "/") for p in install_relpaths
        ]
        relpaths_zip = list(zip(fslash_install_relpaths, install_relpaths))
        del install_relpaths, fslash_install_relpaths

        final_install_relpaths = []

        print("Copying files from CMake output")

        for package_name, relpaths_re in cls.package_paths_re.items():
            package_dest_reldir = package_name.replace(".", os.path.sep)
            for relpath_re in relpaths_re:
                found = False
                r = re.compile(relpath_re + "$")
                for fslash_relpath, relpath in relpaths_zip:
                    m = r.match(fslash_relpath)
                    if not m:
                        continue
                    found = True
                    # We want to keep directory structure as we find it.
                    try:
                        num_dirs_to_remove = len(relpath_re.split('/'))-1
                        new_install_relpath = os.path.join(
                            package_dest_reldir, *pathlib.Path(relpath).parts[num_dirs_to_remove:]
                        )
                        
                    except:
                        new_install_relpath = os.path.join(
                            package_dest_reldir, os.path.basename(relpath)
                        )
                    cls._setuptools_wrap._copy_file(
                        os.path.join(cmake_install_dir, relpath),
                        os.path.join(cmake_install_dir, new_install_relpath),
                        hide_listing=False,
                    )
                    final_install_relpaths.append(new_install_relpath)
                    del m, fslash_relpath, new_install_relpath
                else:
                    if not found:
                        raise Exception("Not found: '%s'" % relpath_re)
                del r, found

        del relpaths_zip

        print("Copying files from non-default sourcetree locations")

        for package_name, paths in cls.files_outside_package.items():
            package_dest_reldir = package_name.replace(".", os.path.sep)
            for path in paths:
                new_install_relpath = os.path.join(
                    package_dest_reldir,
                    # Don't yet have a need to copy
                    # to subdirectories of package dir
                    os.path.basename(path),
                )
                cls._setuptools_wrap._copy_file(
                    path,
                    os.path.join(cmake_install_dir, new_install_relpath),
                    hide_listing=False,
                )
                final_install_relpaths.append(new_install_relpath)

        final_install_paths = [
            os.path.join(cmake_install_dir, p) for p in final_install_relpaths
        ]

        return (cls.wraps._classify_installed_files)(
            final_install_paths,
            package_data,
            package_prefixes,
            py_modules,
            new_py_modules,
            scripts,
            new_scripts,
            data_files,
            # To get around a check that prepends source dir to paths and breaks package detection code.
            cmake_source_dir="",
            cmake_install_dir=cmake_install_reldir,
        )

# This creates a list which is empty but returns a length of 1.
# Should make the wheel a binary distribution and platlib compliant.
class EmptyListWithLength(list):
    def __len__(self):
        return 1


if __name__ == "__main__":
    main()
