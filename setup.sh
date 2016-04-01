#!/usr/bin/env bash
# author: Andre Hollstein
# installs the S2MSI module in a two step process:
#
# 1) Install full module for python3
# 2) Install grass scripts for python2.7 since they do not support python3 jet
#    this assumes that a py27 environment exists which points to a python2 version
#
# The version checking of python is done using the six module.
py27env=py27


function test_alright(){
if [ $? -eq 0 ]
then
  : #-> good exit status
else
  echo "-> bad exit status -> exit here"
  exit 1
fi
}

function install(){
python setup.py clean --all
python setup.py install
test_alright
}

function python_version(){
python -c "from six import PY2; print (PY2)"
}

function uninstall(){
curr_dir=$(pwd)
cd ~
python <<END_OF_PYTHON
#!/usr/bin/env python
from __future__ import print_function
import sys
import inspect
from os.path import dirname
try:
    import S2MSI
    pp = inspect.getabsfile(S2MSI)
    pp = dirname(dirname(pp))
    if pp[-4:] == ".egg":
        print("##################")
        print("You should call:")
        print("rm -rf %s" % pp)
        print("##################")
    else:
        print(pp)

except ImportError as err:
    print("Unable to import package -> nothing to uninstall -> exit here")
    print("Error:")
    print(str(err))

sys.exit(0)
END_OF_PYTHON
cd ${curr_dir}
}

case "$1" in
        "install")
        # install for python above 3.5
        py2=$(python_version)
        if [ "$py2" = "True" ]; then
            echo "It seems that that the Python version is wrong, above 3.5 is needed."
            exit
        fi
        install

        # install for python 2.7
        source activate ${py27env}
        test_alright
        py2=$(python_version)
        if [ "$py2" = "False" ]; then
            echo "It seems that that the Python version is wrong, above 2.7 is needed."
            exit
        fi
        install
        ;;
    "uninstall")
        # uninstall for python above 3.5
        py2=$(python_version)
        if [ "$py2" = "True" ]; then
            echo "It seems that that the Python version is wrong, above 3.5 is needed."
            exit
        fi
        uninstall

        # uninstall for python 2.7
        source activate ${py27env}
        test_alright
        py2=$(python_version)
        if [ "$py2" = "False" ]; then
            echo "It seems that that the Python version is wrong, above 2.7 is needed."
            exit
        fi
        uninstall
        ;;
    *)
        echo "Usage: bash ./setup.sh {install|uninstall}"
        exit 1
        ;;
esac

exit 1
