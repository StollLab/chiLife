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
    style_files = Path("src/chilife/data/mplstyles/").glob("*.mplstyle")
    for style_file in style_files:
        shutil.copy(style_file, mpl_stylelib_path)


class PostInstall(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_style()

setup(cmdclass={'install': PostInstall})

