import setuptools
from setuptools import setup, find_namespace_packages

setup(
    name="aspertinker",
    version="0.1",
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.psi', '*.gjf'],
        # And include any *.msg files found in the 'hello' package, too:
        'aspertinker': ['*.py'],
    },
    package_dir={'': 'namespace/src'},
    #packages=find_namespace_packages(where='src', include=['namespace.*'])
    packages=setuptools.find_packages(where='namespace.src')
)
