# Reusable utilities across projects

Praneeth's tools for making life easy while coding in python. These utilities only depend on packages available through conda or pypi.

## Organization
The tools are organized into the following categories:
Inheritance, Event handlers, File system, Package management, 
Introspection, Input management, Code development,
Communication (with external processes), and Tools for sampled data

## Tool descriptions
Inheritance: (Special cases where I needed to tweak inheritance)  
    * AddMethods      - (Decorator) Add methods to a class
    * Mixin           - (Decorator) Grab methods from another class, and deepcopy list/dict class attributes
    * port_properties - Implement containers with automatic method routing
    * PortProperties  - (Decorator) for using port_properties

Event handlers:  
    * Handler             - Event handlers based on blinker's signal.
    * handler_id2dict     - Turn a handler ID into meaningful parts
    * add_handler         - One-liner access to setting up a broadcaster and receiver.
    * BroadcastProperties - (Decorator) Enables properties in a class to have event handlers.

File system:  
    * locate_command - locate an executable in the system path
    * OnDisk         - (Decorator) Raise error if function output file is not on disk
    * ospath         - Find file or directory
    * find           - Find a file (accepts patterns)
    * run            - Run the contents of a file in the console
    * file_size      - Return size of a list of files in descending order
    * FileManager    - Manage files in a project

Package management: (mostly useful during deployment)  
    * pkg_list - return list of installed packages
    * pkg_path - return path to installed packages

Introspection:  
    * inputs         - Get input variable names and default values of a function
    * module_members - list members of a module
    * properties     - summary of object attributes, properties and methods

Input management:  
    * clean_kwargs - Clean keyword arguments based on default values and aliasing

Code development: (functions that help during code development)  
    * reload  - Reload modules in development folder
    * TimeIt  - (Decorator) Execution time
    * tracker - (decorator) Track objects created by a class (preserves class as class - preferred)
    * Tracker - (Decorator) Track objects created by a class (turns classes into Tracker objects)

Communication:  
    * ExComm         - Communicate with external programs via a socket
    * Spawn          - Use Multiprocessing to run a function in another process (intended for using matplotlib from blender)
    * spawn_commands - Spawn multiple detached processes.

Tools for sampled data:  
    * SampledTime - Encapsulates time and sampling rate
    * Interval    - Start and stop times with extracting samples at different rates
    * SampledData - Encapsulate and manipulate sampled data using signal processing algorithms


## Usage
Create a conda environment with numpy, scipy, multiprocess and blinker  
conda create -n pntools-test python=3.9.2 numpy scipy blinker  
conda activate pntools-test  
conda install -c conda-forge multiprocess  
