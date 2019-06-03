# Installation

### Prerequisites
- Have Python 3.7 installed
- Have the ability to install Python packages

### Install via pip
If you are working in a Linux/Mac environment:
```
sudo pip install ecnet
```

Alternatively, in a Windows or virtualenv environment:
```
pip install ecnet
```

Note: if multiple Python releases are installed on your system (e.g. 2.7 and 3.7), you may need to execute the correct version of pip. For Python 3.7, change **"pip install ecnet"** to **"pip3 install ecnet"**.

To update your version of ECNet to the latest release version, use:
```
pip install --upgrade ecnet
```

### Install from source
Download the ECNet repository, navigate to the download location on the command line/terminal, and execute:
```
python setup.py install
```

Additional package dependencies (ColorLogging, Ditto, ecabc, Keras, NumPy, PubChemPy, PyYaml, TensorFlow) will be installed during the ECNet installation process. If raw performance is your thing, consider building numerical packages like TensorFlow and NumPy from source.
