# use: with cd('...'), returns to the original directory upon encountering exception
# from: http://stackoverflow.com/questions/431684/how-do-i-cd-in-python 

from contextlib import contextmanager
import os

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
