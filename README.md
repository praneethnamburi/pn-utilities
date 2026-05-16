# Reusable utilities across projects

> **This repository is semi-retired.**
>
> As of 2026-05-15, **no active code consumers of `pntools` remain across the
> portfolio**. Every helper that had downstream callers has been migrated to a
> real home — `pysampled`, `datanavigator`, `pyfilemanager`, `datanest`,
> `bpn/utils.py`, or `pn-projects/projects/__init__.py`. The code that remains
> in `pntools/__init__.py` is essentially dead weight; cleaning it up is **not a
> current priority**. The two submodule shims `pntools/sampled.py` and
> `pntools/gui.py` are kept as pickle-compat surfaces (~292 cached pickles
> reference `pntools.sampled.Data` as their `__module__`) and will be retired
> with the portfolio-wide pickle upgrade.
>
> Repository status is therefore **dormant**. New work happens in the
> graduated packages.

Praneeth's tools for making life easy while coding in python. These utilities only depend on packages available through conda or pypi.

## Organization
General tools are in __init__.py and and organized into the following categories:
Inheritance, File system, Package management,
Introspection, Input management, Code development,
Communication (with external processes).

### Submodules

**sampled** (Tools for working with sampled data - refactored into the [pysampled](https://pypi.org/project/pysampled) module):

    * Time      - Encapsulates time and sampling rate
    * Interval  - Start and stop times with extracting samples at different rates
    * Data      - Encapsulate and manipulate sampled data using signal processing algorithms

## Tool descriptions

**Inheritance:** (Special cases where I needed to tweak inheritance)  

    * AddMethods      - (Decorator) Add methods to a class
    * Mixin           - (Decorator) Grab methods from another class, and deepcopy list/dict class attributes

**File system:**  

    * locate_command - locate an executable in the system path
    * OnDisk         - (Decorator) Raise error if function output file is not on disk
    * ospath         - Find file or directory
    * run            - Run the contents of a file in the console

For file-pattern listing, import `pyfilemanager` directly (no longer re-exported here):
`pyfilemanager.FileManager`, `pyfilemanager.find`, `pyfilemanager.get_file_sizes`.

**Package management:** (mostly useful during deployment)  

    * pkg_list - return list of installed packages
    * pkg_path - return path to installed packages

**Introspection:**  

    * inputs         - Get input variable names and default values of a function
    * module_members - list members of a module
    * properties     - summary of object attributes, properties and methods

**Input management:**  

    * clean_kwargs - Clean keyword arguments based on default values and aliasing

**Code development:** (functions that help when developing code)  

    * reload  - Reload modules in development folder
    * TimeIt  - (Decorator) Execution time
    * tracker - (decorator) Track objects created by a class (preserves class as class - preferred)
    * Tracker - (Decorator) Track objects created by a class (turns classes into Tracker objects)

**Communication:**  

    * ExComm         - Communicate with external programs via a socket



## Usage
Create a conda environment using the supplied `environment.yml` file.
If you're unable to import opencv with `import cv2`, then try `pip install opencv-contrib-python`.
