# Bivariate package MUDE
*Version 2.0*

*Authors: Thirza Feenstra, Jelle Knibbe, Irene van der Veer, Robert Lanzafame, Caspar Jungbacker*

Welcome to version 2.0 of the Bivariate package. This README is very incomplete, as it only provides some installation instructions. An example notebook will follow, where the new functionality will be demonstrated.

## Installation
- Recommended: create a fresh conda environment with at least Python 3.10 and PIP:
  
    ```conda create -n bivariate python=3.10 pip```

- Activate the new environment:
  
  ```conda activate bivariate```

- Download the package:
  
  ```https://gitlab.tudelft.nl/mude/bivariate_package/-/raw/main/dist/bivariate-2.0.0-py3-none-any.whl```

- Alternatively, paste these commands in a notebook cell:

  ```from wget import download```
  
  ```download('https://gitlab.tudelft.nl/mude/bivariate_package/-/raw/main/dist/bivariate-2.0.0-py3-none-any.whl')```

  ```!pip install bivariate-2.0.0-py3-none-any.whl```

- Then, install the package using PIP:
  
  ```pip install bivariate-2.0.0-py3-none-any.whl```

If you encounter any issues during installation, please contact me (Caspar), or make a GitLab issue (preferred).

## Working on the package

Robert does the following to test and make updates to the package. Feel free to add suggestions or another section if you do something different:
- fetch and pull from repo
- work in root directory on a notebook (generally `class_test.ipynb`)
- make changes in package files in `bivariate` directory
- restart notebook kernel and rerun cells to check changes
- `add`, `commit`, `push` as needed for package files, notebook, `README.md` (separately)

## Additional information

*Note from Robert on August 28, 2022 (also added to `bivariate notebooks` repo `README.md`):*  
Caspar's example from the `bivariate package` uses `wget` to download and install files. This is easy to install on Mac or Linux OS, but Windows users need to download and install `wget` from [here](https://sourceforge.net/projects/gnuwin32/files/wget/1.11.4-1/wget-1.11.4-1-setup.exe/download?use_mirror=excellmedia), then add the executable directory to the path. [This page](https://techcult.com/how-to-download-install-and-use-wget-for-windows-10/) gives good instructions, except there is a mistake: you should add to the path under 'System variables,' not 'User variables for `user`' (the bottom part of the Environmental Variables window.