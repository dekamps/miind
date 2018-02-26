import imp
import os
import directories

libmiindwc = imp.load_dynamic("libmiindwc", os.path.join(directories.miind_root(), 'build' ,'apps', 'TvbModels', 'wilsoncowan', 'libmiindwc.so'))


