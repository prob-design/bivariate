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


------------------------------
## Guidelines - 06/10/2023 (written by Benjamin Ramousse)

- `src/bivariate` contains the source code for the package
- `tests/test_bivariate.py` is the test file used for automated testing with pytest. In particular, the syntax of the 
methods defined in the said file relates to that of the `pytest` module.
- `tests/examples.ipynb` illustrates the in-use behaviour of the package and the purpose of its methods.

Focus on `src/bivariate/class_multivar.py`:

The rationale behind the creation of the Multivariate class is the construct an object which contains several attributes:
- 3 random variables $X_0, X_1, X_2$ ;
- 2 bivariate copulas $C_{0,1}$ and $C_{1,2}$ between $X_0, X_1$ and $X_1, X_2$, respectively;
- the conditional copula $C_{0,2|1}$.


### Bivariate class
To facilitate the definition of the Multivariate class, a Bivariate class was created. Similarly to Multivariate, 
it contains 3 attributes:
- a list of two random variables (RVs) $[X_0, X_1]$;
- the family of the bivariate copula $C_{0,1}$ (and its parameter);
- the bivariate copula defined using the package `pyvinecopulib` and the couple family/parameter aforementioned.

All the methods of Bivariate access the random variables using their index (0 or 1). This formulation is similarly used
in Multivariate where the index ranges from 0 to 2. The plotting methods are pretty classic, and return f 
(`matplotlib.pyplot.Figure` object) and ax (`matplotlib.pyplot.Axes` object). The `ax` keyword argument present in most 
of Bivariate's plotting methods allows to sequentially add plots on a same Figure object.

### Multivariate class

A Multivariate is defined as follows:

`M = Multivariate([X_0, X_1, X_2], [(family1, parameter1), (family2, parameter2), (family3, parameter3)])`
where family1 (family2) is the family of $C_{0,1}$ ($C_{1,2}$) and parameter1 (parameter2) its parameter. family3 
and parameter3 relate to the conditional copula $C_{0,2|1}$.

Using these arguments, two Bivariate objects are created for $X_0, X_1$ and $X_1, X_2$. The conditional copula and 
the two bivariate copulas are then used to sample the copula $C_{0,2}$ of $X_0, X_2$. 

**Note**: the current version only applies if all the (conditional) copulas are normal. Other special cases (and treatment of the sampling) should be implemented for generalization.

A key method of the Multivariate class is `bivariate_plot` which allows to plot a (limit-state) function and the 
multivariate joint distribution's contours in a given bivariate plan. `x_index` and `y_index` are the indices of the 
variables taken for the plot's x and y axes: for instance, `x_index`=1 and `y_index`=0 positions the plot in the plane 
$(x_1, x_0)$. `z_value` is the value at which the third (not plotted) random variable is set: in the previous example
in the plane $(x_1, x_0)$, `z_value` allows to set the value of $X_2$ used for the plot. 

The same `x_index` and `y_index` system is used in the other plotting methods of the class.


### Structure of bivariate package, Siemen

.
├── github
├── src\bivariate                  # 
│   ├── __init__.py                # Documentation files (alternatively `doc`)
│   ├── class_dataset.py           # Source files (alternatively `lib` or `app`)
│   ├── class_multivar.py          # Source files (alternatively `lib` or `app`)
|        |-- class_bivariate.py    
├── ...                     # Source files (alternatively `lib` or `app`)
└── README.md

