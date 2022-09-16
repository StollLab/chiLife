from setuptools import setup
from pathlib import Path
import shutil

setup(
    name='chiLife',
    version='0.1',
    packages=['chiLife'],
    package_data={'chiLife': ['data/*', 'data/*/*', 'data/*/*/*']},
    url='',
    license='',
    author='Maxx Tessmer',
    author_email='mhtessmer@gmail.com',
    install_requires=['numpy>=1.23.0',
                      'scipy>=1.6.3',
                      'matplotlib>=3.3.4',
                      'numba>=0.56.2',
                      'mdanalysis>=2.0.0',
                      'tqdm>=4.45.0',
                      'pytest>=6.2.2',
                      'memoization>=0.3.1',
                      'argparse>=1.4.0',
                      'setuptools>=53.0.0'
                      'networkx>=2.8'],
    description=''
)

import matplotlib as mpl
mlp_stylelib_path = Path(mpl.get_data_path(), 'stylelib')
style_files = Path("mplstyles/").glob("*.mplstyle")
for style_file in style_files:
    shutil.copy(style_file, mlp_stylelib_path)
