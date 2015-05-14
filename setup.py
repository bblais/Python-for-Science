from __future__ import with_statement

from distutils.core import setup

import numpy

def get_version():
    
    d={}
    version_line=''
    with open('science/__init__.py') as fid:
        for line in fid:
            if line.startswith('__version__'):
                version_line=line
    print version_line
    
    exec(version_line,d)
    return d['__version__']
    

setup(
  name = 'science',
  version=get_version(),
  description="Utilities for Science",
  author="Brian Blais",
  packages=['science'],
  package_data      = {'science': ['science.mplstyle']},
)


