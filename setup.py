from setuptools import setup
from setuptools.command.install import install
from pathlib import Path
import shutil

with open('README.md', 'r', encoding='utf8') as f:
    readme = f.read()


def install_style():
    import matplotlib as mpl
    mpl_stylelib_path = Path(mpl.get_configdir()) / 'stylelib'
    mpl_stylelib_path.mkdir(parents=True, exist_ok=True)
    style_files = Path("chilife/data/mplstyles/").glob("*.mplstyle")
    for style_file in style_files:
        shutil.copy(style_file, mpl_stylelib_path)


class PostInstall(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_style()

setup(
    name='chilife',
    packages=['chilife'],
    package_data={'chilife': ['data/*', 'data/*/*', 'data/*/*/*']},
    scripts=['scripts/update_rotlib.py', 'scripts/oldProteinIC.py'],
    project_urls = {'Source': 'https://github.com/StollLab/chiLife'},
    long_description=readme,
    long_description_content_type='text/markdown',
    cmdclass={'install': PostInstall})

