from setuptools import Extension, setup

module = Extension("spkmeans",
                  sources=[
                    'spkmeans.c',
                    'spkmeansmodule.c'
                  ])
setup(name='spkmeans',
     version='1.0',
     description='Python wrapper for spectral clustering algorithm '
                 'implemented in C',
     ext_modules=[module])
