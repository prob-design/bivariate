# Bivariate package MUDE
*Version 2.0*

*Authors: Thirza Feenstra, Jelle Knibbe, Irene van der Veer, Robert Lanzafame, Caspar Jungbacker*

Welcome to version 2.0 of the Bivariate package. This README is very incomplete, as it only provides some installation instructions. An example notebook will follow, where the new functionality will be demonstrated.

This package is created to support teachers and teaching assistance in the course Modeling, Uncertainty and Data for Engineers at Delft University of Technology. Students are welcome to use the package if interested, although we ask that you don't use it to solve problems in MUDE assignments unless explicitly asked to do so, or if you want to use it to study in your free time.

## Installation

*Make sure you have the Python module `wget` installed (from terminal: `pip install wget`). If the repo does not stay public, the wheel file should be downloaded manually from GitLab.*  
Paste these commands in a notebook cell:  
```
from wget import download  
download('https://gitlab.tudelft.nl/mude/bivariate_package/-/raw/main/dist/bivariate-2.0.0-py3-none-any.whl')  
!pip install bivariate-2.0.0-py3-none-any.whl
```

Technically you can also create and activate a fresh conda environment with at least Python 3.10 and `pip`, prior to running the above, with these two commands:  
```
conda create -n bivariate python=3.10 pip  
conda activate bivariate  
```


## Updating the package

*A future version will automatically regenerate the wheel file (Issue #15).*

### Reinstall on `PATH`

If you installed the package as instructed above, the following command will update your local installation (first two commands are same as original installation). Make sure you check two things:
- the wheel file on GitLab has been updated since your most recent (re)installation
- the wheel file is not already in your working directory (`wget` will not overwrite the old file)  

```
from wget import download  
download('https://gitlab.tudelft.nl/mude/bivariate_package/-/raw/main/dist/bivariate-2.0.0-py3-none-any.whl')  
!pip install --force-reinstall --no-deps bivariate-2.0.0-py3-none-any.whl
```

### Update in local working directory

Copy the directory `bivariate` to your local working directory and import as:
```from bivariate.class_dataset import Dataset```

This is a more detailed description of a working method when you are actively making changes to the package and using those changes in another repo:  
- fetch and pull from `bivariate` repo
- copy bivariate package/bivariate to local directory where you have your notebook  
  - example: from terminal use `cp -r ../'bivariate package'/bivariate .`  
  - example assumes you are working in a notebook in a directory that is 'parallel' to your bivariate package repo  
- work in root directory, typically on a notebook (generally `class_test.ipynb`)
- use `from bivariate.class_dataset import Dataset` to import package in notebook 
- make changes in package files in `bivariate` directory
- restart notebook kernel and rerun cells to check changes (*for better solution, see Issue #12*)
- `add`, `commit`, `push` as needed for package files, notebook, `README.md` (separately)

## Update the wheel file

To update the package distribution (wheel file, `*.whl`) run the following from the terminal and push the changes:
```python setup.py bdist_wheel```
