Application with Deep Learning
==============================

Machine Configuration
---------------------

    Deep learning is a field with intense computational requirements mainly when you deal with image processing.

    For this project we proposed a structure to integrate GEE (Google Earth Engine) and Tensorflow.

1. Minimal Requirements

    | Google Cloud Account
    | Create an computer engine instance:
    |   1 core
    |   3.5gb RAM
    |   60gb of Storage (hdd or ssd)
    |   Ubuntu 16.04
    |   1 GPU Nvidia K80

2. Installation of the libraries

    This material is based on `Using a GPU & TensorFlow on Google Cloud Platform <https://medium.com/google-cloud/using-a-gpu-tensorflow-on-google-cloud-platform-1a2458f42b0>`_

    - Install python and jupyter

    .. code-block:: bash

        # system update
        apt-get -y update

        # python pip installation
        apt install -y python-pip

        # config to notebook
        apt-get -y install ipython

        apt-get -y install ipython-notebook

        # jupyter installation
        pip install jupyter

        apt-get -y install python3-pip

        python3 -m pip install ipykernel

        # python kernel to jupyter notebook
        python3 -m ipykernel install --user

        # only python 3
        cd /notebooks && jupyter notebook --allow-root --ip='*'

3. Driver installation

    In this point it is required to install `nvidia driver <https://www.nvidia.com.br/Download/driverResults.aspx/135486/br>`_ and `gpu driver <http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/>`_.

    To install the gpu driver it can be used this code

    .. code-block:: bash

        #!/bin/bash
        echo "Checking for CUDA and installing."
        # Check for CUDA and try to install.
        if ! dpkg-query -W cuda; then
            # The 16.04 installer works with 16.10.
            curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
            dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
            apt-get update
            # apt-get install cuda -y
            sudo apt-get install cuda-9-0
        fi

4. Installing cuDNN

    In this point it is needed to be logged in nvidia website and download `cudnn-download <https://developer.nvidia.com/rdp/cudnn-download>`_ and send to the server by scp copy file.

    To install:

    .. code-block:: bash

        cd $HOME
        tar xzvf cudnn-9.0-linux-x64-v5.1.tgz
        sudo cp cuda/lib64/* /usr/local/cuda/lib64/
        sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
        rm -rf ~/cuda
        rm cudnn-9.0-linux-x64-v5.1.tgz

5. Test

    Install library on system

    .. code-block:: bash

        # must use tensorflow 1.9
        pip install --upgrade tensorflow-gpu==1.9.0

        nvidia-smi -l

    Run code

    .. code-block:: python

        # tensorflow library
        import tensorflow as tf
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))