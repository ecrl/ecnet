from setuptools import setup, find_packages
from platform import system, mac_ver

install_requires = [
    'alvadescpy==0.1.0',
    'colorlogging==0.3.5',
    'ecabc==2.2.3',
    'matplotlib==3.1.2',
    'numpy==1.16.4',
    'padelpy==0.1.6',
    'pyyaml==5.1.1',
    'scikit-learn==0.21.2'
]

s = system()
if s == 'Windows':
    install_requires.append(
        'torch @ https://download.pytorch.org/whl/cpu/torch-1.3.1%2Bcpu-cp37-cp37m-win_amd64.whl'
    )
elif s == 'Linux':
    if mac_ver()[0] != '':
        install_requires.append(
            'torch @ https://download.pytorch.org/whl/cpu/torch-1.3.1-cp37-none-macosx_10_7_x86_64.whl'
        )
    else:
        install_requires.append(
            'torch @ https://download.pytorch.org/whl/cpu/torch-1.3.1%2Bcpu-cp37-cp37m-linux_x86_64.whl'
        )
else:
    raise OSError('Unsupported OS: {}'.format(s))

setup(
    name='ecnet',
    version='3.2.3',
    description='UMass Lowell Energy and Combustion Research Laboratory Neural'
                ' Network Software',
    url='https://github.com/ecrl/ecnet',
    author='Travis Kessler, John Hunter Mack',
    author_email='Travis_Kessler@student.uml.edu,'
                 ' Hunter_Mack@uml.edu',
    license='MIT',
    packages=find_packages(),
    python_requires='~=3.7',
    install_requires=install_requires,
    zip_safe=False
)
