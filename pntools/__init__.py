"""
Praneeth's tools for making life easy while coding in python.
"""

import errno
import fnmatch
import functools
import importlib
import inspect
import os
import re
import socket
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd

import numpy as np

try:
    BLENDER_MODE = True
    import bpy
except ImportError:
    BLENDER_MODE = False

import pyfilemanager


## Inheritance
class AddMethods:
    """
    Add methods to a class. Decorator.
    
    Usage:
        @AddMethods([pntools.properties])
        class myClass:
            def foo(self):
                print("bar")
        
        a = myClass()
        a.foo() # prints bar
        a.properties() # lists foo and properties
    """
    def __init__(self, methodList):
        self.methodList = methodList
    def __call__(self, func):
        functools.update_wrapper(self, func)
        @functools.wraps(func)
        def wrapperFunc(*args, **kwargs):
            for method in self.methodList:
                setattr(func, method.__name__, method)
            funcOut = func(*args, **kwargs)
            return funcOut
        return wrapperFunc

class MixIn:
    """
    Decorator to grab properties from src class and put them in target class.
    No overwrites.
    ALMOST the same as inheritance, BUT, if the source class has a
    'list' or 'dict' class attribute, then it makes a deepcopy
    Example:
        @MixIn(src_class)
        class trg_class:
            pass
    """
    def __init__(self, src_class):
        self.src_class = src_class
    def __call__(self, trg_class):
        src_attrs = {attr_name:attr for attr_name, attr in self.src_class.__dict__.items() if attr_name[0] != '_'}
        for src_attr_name, src_attr in src_attrs.items():
            if not hasattr(trg_class, src_attr_name): # no overwrites
                if isinstance(src_attr, (list, dict)):
                    src_attr = deepcopy(src_attr)
                setattr(trg_class, src_attr_name, src_attr)
        return trg_class

def port_properties(src_class, trg_class, trg_attr_name='data'):
    """
    Port properties and methods (not hidden) from source class src_class
    to target class trg_class.

    Differs from Mixin and inheriance. Used to design 'containers' with automatic routing.

    :param src_class: (class)
    :param trg_class: (class)
    :param trg_attr_name: (str)

    Basically, trg_class objects have an attribute with name
    trg_attr_name, which is an instance of trg_class
    Example:
        MeshObject class has an attribute (or property) 'data' that is an instance of trg class
        s = MeshObject() # MeshObject is the trg_class
        s.data = Mesh()  # Mesh is the src_class
    Now,
        s.data.prop : to execute this, I want to say s.prop
        s.data.func() : to execute this, I want to say s.func()
        
    Within MeshObject, defining the __init__ function as follows achieves this!
    class MeshObject(Object):
        def __init__(self, name, *args, **kwargs):
            super().__init__(name, *args, **kwargs) # make an instance of Object
            self.data = Mesh(...) # make an instance of mesh
            pn.port_properties(Mesh, self.__class__, 'data') 
            # grants direct access to Mesh's stuff with appropritate routing

    Now a MeshObject instance inherits all methods and properties from
    Object class, AND from Mesh class. Methods from Mesh class are
    automagically called with the correct Mesh object as the first input.

    Note that trg_class itself is being modified
    (i.e., return statement is just to enable the decorator)

    Note that attributes of the Mesh class will NOT be copied
    """
    # properties
    def swap_input_fget(this_prop):
        return lambda x: this_prop.fget(getattr(x, trg_attr_name))
    
    def swap_input_fset(this_prop):
        return lambda x, s: this_prop.fset(getattr(x, trg_attr_name), s)

    src_properties = {p_name : p for p_name, p in src_class.__dict__.items() if isinstance(p, property)}
    for p_name, p in src_properties.items():
        if not hasattr(trg_class, p_name): # no overwrites - this implmentation is more readable
            if p.fset is None:
                setattr(trg_class, p_name, property(swap_input_fget(p)))
            else:
                setattr(trg_class, p_name, property(swap_input_fget(p), swap_input_fset(p)))

    # methods
    def swap_first_input(func): # when we don't know how many inputs func has
        return lambda x: functools.partial(func, getattr(x, trg_attr_name))

    src_methods = {func_name:func for func_name, func in src_class.__dict__.items() if type(func).__name__ == 'function' and func_name[0] != '_'}
    for src_func_name, src_func in src_methods.items():
        if not hasattr(trg_class, src_func_name): # no overwrites
            setattr(trg_class, src_func_name, property(swap_first_input(src_func)))

    return trg_class

class PortProperties:
    """
    Providing port_properties functionality as a decorator.
    
    This is for implementing the idea of a 'container' in blender, that
    I could not solve using multiple inheritance. A container is an
    instance of a specific class, but also contains instances of other
    classes. You can act on any 'contained' object directly (you just
    have to use the method name)
    cont = primary_class()
    cont.two = secondary_class()
    If dummy is a method of cont.two, then I want to say:
    cont.dummy() instead of cont.two.dummy(), 
    BUT cont.dummy() should execute cont.two.dummy() if cont doesn't have a method called dummy()

    A container class is created by:
    Inheriting from a primary class.
    Modifiying the container class with port_properties (or using the PortProperties decorator)

    Example:
    @PortProperties(Mesh, 'data') # instance of MeshObject MUST have 'data' attribute/property
    class MeshObject(Object):   
        def __init__(self):
            super().__init__()
            self.data = Mesh()

    m = MeshObject()

    In this example, MeshObject is the container class.
    It inherits from Object class.
    Mesh is the secondary class
    """
    def __init__(self, src_class, trg_attr_name):
        self.src_class = src_class
        self.trg_attr_name = trg_attr_name
    def __call__(self, trg_class):
        return port_properties(self.src_class, trg_class, self.trg_attr_name)


## File system
class OnDisk:
    """
    Raise error if function output not on disk. Decorator.

    :param func: function that outputs a path/directory
    :returns: decorated function with output handling
    :raises keyError: FileNotFoundError if func's output is not on disk

    Example:
        @OnDisk
        def getFileName_full(fPath, fName):
            fullName = os.path.join(os.path.normpath(fPath), fName)
            return fullName
    """
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)
    def __call__(self, *args, **kwargs):
        thisDirFiles = self.func(*args, **kwargs)
        self.checkFiles(thisDirFiles)
        return thisDirFiles
    @staticmethod
    def checkFiles(thisFileList):
        """Raise error if any file in thisFileList not on disk."""
        if isinstance(thisFileList, str):
            thisFileList = [thisFileList]
        for dirFile in thisFileList:
            if not os.path.exists(dirFile):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dirFile)

def change_image_dpi(files, dpi:int=300, return_format:str='tif'):
    """
    Change the dpi of a set of images, example - for publication

    Usage -
        file_list = pyfilemanager.find('*.tif', r'C:\Dropbox (MIT)\Manuscripts\20230401 Elastic energy EDM\Premiere pro')
        change_image_dpi(file_list)
    """
    from PIL import Image

    if isinstance(files, str):
        if os.path.isdir(files):
            file_list = pyfilemanager.find('*.tif', path=files)
        else:
            file_list = [files]
    else:
        assert isinstance(files, (list, tuple))
        file_list = files

    for fname in file_list:
        if not str(Path(fname).stem).endswith(f'_{dpi}dpi'):
            im = Image.open(fname)
            im.save(str(Path(fname).with_suffix(''))+f'_{dpi}dpi.{return_format}', dpi=(dpi,dpi))

def run(filename, start_line=1, end_line=None):
    """
    Most commonly used to run code that I'm working on inside the console.
    NOTE MATLAB-like indexing run(x.py, 1, 2) runs lines 1 and 2
    That the line numbers are 1-indexed to match the line numbers in the code editor (VSCode)
    Runs the last line number indicated as well!
    """
    if not os.path.isfile(filename):
        filename = pyfilemanager.find(filename)[0]
    assert os.path.isfile(filename)
    code = open(filename).readlines()
    if end_line is None:
        end_line = len(code)
    exec(''.join(code[(start_line-1):end_line]))


## Package management
def pkg_list():
    """
    Return a list of installed packages.

    :returns: output of pip freeze
    :raises keyError: raises an exception
    """
    proc = subprocess.Popen('pip freeze', stdout=subprocess.PIPE, shell=True)
    out = proc.communicate()
    pkgs = out[0].decode('utf-8').rstrip('\n').split('\n')
    pkgs = [k.rstrip('\r') for k in pkgs]  # windows compatibility
    pkgNames = [m[0] for m in [k.split('==') for k in pkgs]]
    pkgVers = [m[0] for m in [k.split('==') for k in pkgs]]
    return pkgs, pkgNames, pkgVers

def pkg_path(pkgNames=None):
    """Return path to installed packages."""
    if not pkgNames:
        _, pkgNames, _ = pkg_list()
    elif isinstance(pkgNames, str):
        pkgNames = [pkgNames]
    
    currPkgDir = []
    failedPackages = []
    for pkgName in pkgNames:
        print(pkgName)
        if pkgName == 'ipython':
            pkgName = 'IPython'
        elif pkgName == 'ipython-genutils':
            pkgName = str(pkgName).lower().replace('-', '_')
        elif pkgName in ['pywinpty', 'pyzmq', 'terminado']:
            continue
        else:
            pkgName = str(pkgName).lower().replace('-', '_').replace('python_', '').replace('_websupport', '')
        try:
            currPkgDir.append(importlib.import_module(pkgName).__file__)
        except UserWarning:
            failedPackages.append(pkgName)

    print('Failed for: ', failedPackages)    
    return currPkgDir


## introspection
def inputs(func):
    """Get the input variable names and default values to a function."""
    inpdict = {}
    if callable(func):
        inpdict = {str(k):inspect.signature(func).parameters[str(k)].default for k in inspect.signature(func).parameters.keys()}
    return inpdict

def properties(obj):
    """
    For an instance obj of any class, use pn.properties(obj) for a summary of properties.
    Especially useful in the blender console.
    """
    #pylint:disable=expression-not-assigned
    [print((k, type(getattr(obj, k)), np.shape(getattr(obj, k)))) for k in dir(obj) if '_' not in k and 'method' not in k]


## Code development
class TimeIt:
    """
    Prints execution time. Decorator.
    Note that this only works on functions.
    Consider a function call:
    out1 = m.inflate(0.15, 0.1, 0.02, 100)
    Using the following will give the output in addition to printing the
    execution time.
    out1 = pn.TimeIt(m.inflate)(0.15, 0.1, 0.02, 100)
    """
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)
    def __call__(self, *args, **kwargs):
        start = timer()
        funcOut = self.func(*args, **kwargs)
        end = timer()
        print(end-start)
        return funcOut

def tracker(trg_class):
    """
    Use this as a decorator to track class instances (and keep tracked classes as classes).
    include self.track(self) in the decorated class' __init__
    If there is a tracker in the parent class, don't add self.track(self) to the child class.
    BUT, decorate the child class!!
    
    Example:
        @tracker
        class Thing:
            def __init__(self):
                self.track(self)

        @tracker
        class Mesh:
            def __init__(self):
                pass # Don't track again!
    """
    class TrackMethods:
        """
        This is just a method container.
        see tracker function
        """
        all = []
        cache = []

        @classmethod
        def track(cls, obj):
            """Just used by the initalization function to track object."""
            cls.all.append(obj)
        
        @classmethod
        def track_clear(cls):
            """Forget the objects tracked so far."""
            cls.all = []
        
        @classmethod
        def track_clear_cache(cls):
            """Clear cache used by temporary tracking sessions."""
            cls.cache = []

        @classmethod
        def track_start(cls):
            """
            Start a tracking session. Move current objects to cache, and clean.
            Note that objects are tracked even without this method if a class is being tracked.
            Use this to create temporary tracking sessions.
            """
            cls.cache = cls.all
            cls.all = []

        @classmethod
        def track_end(cls):
            """End tracking session."""
            cls.all = cls.cache
            cls.cache = []

        @classmethod
        def dict_access(cls, key='id', val=None):
            """
            Give access to the object based on key. 
            
            Note:
            If keys (id) of different objects are the same, then only the
            last reference will be preserved.

            :param key: Property of the object being tracked (to be used as the key).
            :param val: Property of the object being tracked (to be used as the value).
                        When set to None, val is set to the object itself.
            :returns: A dictionary of property pairs for all objects in key(property1):val(property2)
            """
            if not val:
                return {getattr(k, key):k for k in cls.all}
            
            return {getattr(k, key):getattr(k, val) for k in cls.all}

    src_class = TrackMethods
    src_attrs = {attr_name:attr for attr_name, attr in src_class.__dict__.items() if attr_name[0] != '_'}
    # deliberately overwrite all and cache
    trg_class.all = deepcopy(src_class.all)
    trg_class.cache = deepcopy(src_class.cache)
    for src_attr_name, src_attr in src_attrs.items():
        if not hasattr(trg_class, src_attr_name): # no overwrites
            setattr(trg_class, src_attr_name, src_attr)
    return trg_class

class Tracker:
    """
    Keep track of all instances of objects created by a class.
    This converts a class into a Tracker object. 
    To keep a class as a class, decorate with the tracker function.
    clsToTrack. Decorator. 

    Meant to convert clsToTrack into a Tracker object with properties
    all, n, and methods dictAccess

    TODO:list 
    1. keeping track of all tracked classes is controversial because all
       the tracked objects (formerly classes) know what other classes
       are being tracked. (see cls._tracked)

    :param clsToTrack: keep track of objects created by this class
    :returns: a tracker object

    For operations that span across all objects created by a clsToTrack,
    you can simply create a groupOperations class without __new__,
    __init__ or __call__ functions, and decorate clsToDecorate with that
    class. See tests for an example.

    Example of extending the Tracker class:
    class ImgGroupOps(my.Tracker):
        def __init__(self, clsToTrack):
            super().__init__(clsToTrack)
            self.load()
        
        def load(self):
    """
    _tracked = [] # all the classes being tracked
    def __new__(cls, clsToTrack):
        cls._tracked.append(clsToTrack)
        return object.__new__(cls)
    def __init__(self, clsToTrack):
        self.clsToTrack = clsToTrack
        functools.update_wrapper(self, clsToTrack)
        self.all = []
        self.cache = []
    def __call__(self, *args, **kwargs):
        funcOut = self.clsToTrack(*args, **kwargs)
        self.all.append(funcOut)
        return funcOut
    def __delitem__(self, item):
        """
        Stop tracking item.
        item is an instance of clsToTrack
        """
        self.all.remove(item)

    def dictAccess(self, key='id', val=None):
        """
        Give access to the object based on key. 
        
        Note:
        If keys (id) of different objects are the same, then only the
        last reference will be preserved.

        :param key: Property of the object being tracked (to be used as the key).
        :param val: Property of the object being tracked (to be used as the value).
                    When set to None, val is set to the object itself.
        :returns: A dictionary of property pairs for all objects in key(property1):val(property2)
        """
        if not val:
            return {getattr(k, key):k for k in self.all}
        
        return {getattr(k, key):getattr(k, val) for k in self.all}

    @property
    def n(self):
        """Return the number of instances being tracked."""
        return len(self.all)

    def query(self, queryStr="agent == 'sausage' and accuracy > 0.7", keys=None):
        """
        Filter all tracked objects based on object fields (keys).
        
        :param queryStr: list-comprehension style filter string
        :param keys: list of object keys used in the query
        :returns: a subset of tracked objects that satisfy query criteria
        :raises Warning: prints processed query string if the query fails

        Refer to tests for examples and notes on how to use.
        """
        if not queryStr:
            return self.all

        def parseQuery(queryStr):
            queryUnits = re.split(r'and|or', queryStr)
            queryUnits = [queryUnit.lstrip(' ').lstrip('(').lstrip(' ') for queryUnit in queryUnits]
            for queryUnit in queryUnits:
                if ' in ' not in queryUnit:
                    queryStr = queryStr.replace(queryUnit, 'k.'+queryUnit)

            queryStr = queryStr.replace(' in ', ' in k.')
            return queryStr

        if keys is None:
            queryStr = parseQuery(queryStr)
        else:
            for key in keys:
                queryStr = queryStr.replace(key, 'k.'+key)

        try:
            objList = eval("[k for k in self.all if " + queryStr + "]") #pylint:disable=eval-used
        except Warning:
            print('Query failed.')
            print(queryStr)
        return objList

    def clean(self):
        """Forget the objects tracked so far."""
        self.all = []
    
    def clear_cache(self):
        """Clear cache used by temporary tracking sessions."""
        self.cache = []

    def track_start(self):
        """
        Start a tracking session. Move current objects to cache, and clean.
        Note that objects are tracked even without this method if a class is being tracked.
        Use this to create temporary tracking sessions.
        """
        self.cache = self.all
        self.clean()

    def track_end(self):
        """End tracking session."""
        self.all = self.cache
        self.clear_cache()


class ExComm:
    """
    For communicating with other programs via TCPIP.
    For MATLAB communication, use MATLAB engine!
    """
    def __init__(self, host='localhost', port=50000):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        print("waiting for response from client at port ", port)
        self.host = host
        self.port = port
        self.conn, self.addr = s.accept()
        print('Connected by', self.addr)

    def __pos__(self):
        # listen
        data = self.conn.recv(1024)
        print(data)

    def __call__(self, message=b"hello"):
        # send data
        self.conn.sendall(message)
    
    def __neg__(self):
        # close connection
        self.conn.close()

def spawn_commands(cmds, nproc=3, verbose=False, retry=False, sleep_time=0.5, wait=True):
    """
    Spawn multiple detached processes. Originally designed for converting videos using ffmpeg.
    cmds is a list of commands, and each command is a list that can be supplied to subprocess.Popen
    """
    n_running = lambda: sum([int(p.poll() is None) for p in all_proc])
    all_proc = []
    cmd_count = 0
    if nproc > len(cmds):
        nproc = len(cmds)

    while True:
        if n_running() < nproc and cmd_count < len(cmds):
            if os.name == 'nt':
                # CREATE_NO_WINDOW = 0x08000000: suppress the console window for child ffmpeg etc.
                # shell=False lets Popen quote argv entries containing spaces correctly on Windows.
                # creationflags=0x00000008 will spawn in a new window
                all_proc.append(subprocess.Popen(cmds[cmd_count], creationflags=0x08000000))
            else:
                all_proc.append(subprocess.Popen(cmds[cmd_count], stderr=subprocess.STDOUT, stdout=subprocess.PIPE))
            time.sleep(sleep_time)
            if all_proc[-1].poll() == 1 and retry:
                # process exited - probably graphics card out of memory
                all_proc.pop()
                if nproc > 1:
                    nproc -= 1
            else:
                cmd_count += 1
            if verbose:
                print({'Poll': [p.poll() for p in all_proc], 'Running': n_running()})
            if cmd_count == len(cmds):
                break
    
    if wait:
        while n_running() > 0:
            time.sleep(sleep_time)

    return all_proc


## extensions to basic classes
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class namelist:
    """List of elements where each element has the 'name' field"""
    def __init__(self, data):
        self.data = list(data)
    
    @property
    def names(self):
        return [x.name for x in self.data]
    
    def __getitem__(self, key):
        if key not in self.names:
            if isinstance(key, int):
                return self.data[key]
            else:
                print(self.names)
                raise KeyError
        return {d.name : d for d in self.data}[key]

class nameidlist(namelist):
    """List of elements where each element has 'name' AND 'id' fields."""
    @property
    def ids(self):
        return [x.id for x in self.data]

    def __call__(self, key=None):
        if key is None:
            print(self.ids)
            return
        return {x.id:x for x in self.data}[key]

def flip_nested_dict(data:dict) -> dict:
    """Flip the hierarchy in a nested dictionary"""
    flipped = {}
    for key, val in data.items():
        for subkey, subval in val.items():
            if subkey not in flipped:
                flipped[subkey] = {}
            flipped[subkey][key] = subval
    return flipped

## Simple operations
def split_filename(fname:str) -> tuple:
    name, ext = os.path.splitext(os.path.basename(fname))
    path = os.path.dirname(fname)
    return path, name, ext

class Mapping:
    """Create a dictionary map between any two columns of a dataframe"""
    def __init__(self, df:pd.DataFrame):
        self.df = df
    
    def __call__(self, left_col_name:str, right_col_name:str, row_selector=None) -> dict:
        if row_selector is None:
            row_selector = lambda k,v: True
        ret = pd.Series(self.df[right_col_name].values,index=self.df[left_col_name]).to_dict()
        return {k:v for k,v in ret.items() if row_selector(k,v)}
    
def is_numeric(s:str) -> bool:
    """Is a string numeric"""
    assert isinstance(s, str)
    return s.removeprefix('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()

def to_number(s:str):
    """If string is a number, return the number, otherwise return the original string"""
    if not isinstance(s, str):
        return s
    if is_numeric(s):
        return eval(s.lstrip('0'))
    return s

# wrapper around dateutil to check if a string is or has a date
try: # YET to be tested thoroughly
    from dateutil.parser import parse
    def is_date(s:str) -> bool:
        """Check is string is a date"""
        try: 
            parse(s, fuzzy=False)
            return True
        except ValueError:
            return False
    
    def has_date(s:str) -> bool:
        """Check is string is a date"""
        try: 
            parse(s, fuzzy=True)
            return True
        except ValueError:
            return False
    
    def to_date(s:str, strict=False):
        if not isinstance(s, str):
            return s
        if strict:
            if is_date(s):
                return parse(s, fuzzy=False)
            return s

        if has_date(s):
            if is_date(s):
                return parse(s, fuzzy=False)
            return parse(s, fuzzy_with_tokens=True)[0]
        return s # doesn't have a date, function doesn't do any transform

    def to_date_or_number(s:str):
        return to_date(to_number(s), strict=True)
    
except (ModuleNotFoundError, ImportError):
        raise('Install python-dateutil to use pntools.is_date. e.g. pip install python-dateutil')


try:
    import portion as P

    # extend portion functionality in the class below
    class PNInterval(P.Interval):
        @property
        def atomic_durations(self):
            return [xi.upper - xi.lower for xi in self]
        
        @property
        def duration(self):
            return sum(self.atomic_durations)
        
        @property
        def fraction(self):
            # fractional duration relative to the enclosure
            return self.duration/self.enclosure.duration
        
    portion = P.create_api(PNInterval)
except ModuleNotFoundError:
    print('portion is not installed in this environment. conda install -c conda-forge portion.')

def apply_to_files(file_selectors, include=(), exclude=(), ret_type=list):
    """Decorator for functions/classes whose first argument is a file name. 
    For these functions/classes, if a folder is supplied instead of a file name, 
    the decorator will find all the relevant files and apply the decorated function to each file.
    Apply func to all files in a folder."""
    if isinstance(file_selectors, str):
        file_selectors = [file_selectors]
    assert ret_type in (list, dict, 'stem') # stem returns a dict with the keys as stems of the filenames as opposed to the full file name
    def wrapper(func):
        def inner_func(fname, *args, **kwargs):
            if os.path.isdir(fname):
                fm = pyfilemanager.FileManager(fname)
                for file_selector_count, file_selector in enumerate(file_selectors):
                    fm.add(str(file_selector_count), file_selector, include=include, exclude=exclude)
                if ret_type == list:
                    return [func(file_name, *args, **kwargs) for file_name in fm.all_files]
                if ret_type == 'stem':
                    return {Path(file_name).stem:func(file_name, *args, **kwargs) for file_name in fm.all_files}
                assert ret_type == dict
                return {file_name:func(file_name, *args, **kwargs) for file_name in fm.all_files}
            else:
                return func(fname, *args, **kwargs)
        return inner_func
    return wrapper

@apply_to_files('*.*')
def test_apply_to_files(fname):
    print(fname)

try:
    from outliers import smirnov_grubbs as grubbs
    class Grubbs(grubbs.TwoSidedGrubbsTest):
        def __init__(self, data, alpha=0.05):
            super().__init__(data)
            self.alpha = alpha

        @property
        def critical_value(self):
            return self._get_g_test(self._copy_data(), alpha=self.alpha)
        
        @property
        def data_without_outliers(self):
            return self.run(alpha=self.alpha, output_type=grubbs.OutputType.DATA)
        
        @property
        def outlier_values(self):
            return self.run(alpha=self.alpha, output_type=grubbs.OutputType.OUTLIERS)
        
        @property
        def outlier_indices(self):
            return self.run(alpha=self.alpha, output_type=grubbs.OutputType.INDICES)
        
        @property
        def z_values(self):
            return np.abs(self.original_data-self.original_data.mean())/self.data_without_outliers.std()
        
except ModuleNotFoundError:
    print('outlier_utils is not installed in this environment. pip install outlier_utils.')

"""
Implimentation of Density-Based Clustering Validation "DBCV"

Citation:
Moulavi, Davoud, et al. "Density-based clustering validation."
Proceedings of the 2014 SIAM International Conference on Data Mining.
Society for Industrial and Applied Mathematics, 2014.
"""

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist, euclidean


def DBCV(X, labels, dist_function=euclidean):
    """
    Density Based clustering validation

    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point

    Returns: cluster_validity (float)
        score in range[-1, 1] indicating validity of clustering assignments
    """
    graph = _mutual_reach_dist_graph(X, labels, dist_function)
    mst = _mutual_reach_dist_MST(graph)
    cluster_validity = _clustering_validity_index(mst, labels)
    return cluster_validity


def _core_dist(point, neighbors, dist_function):
    """
    Computes the core distance of a point.
    Core distance is the inverse density of an object.

    Args:
        point (np.array): array of dimensions (n_features,)
            point to compute core distance of
        neighbors (np.ndarray): array of dimensions (n_neighbors, n_features):
            array of all other points in object class
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point

    Returns: core_dist (float)
        inverse density of point
    """
    n_features = np.shape(point)[0]
    n_neighbors = np.shape(neighbors)[0]

    distance_vector = cdist(point.reshape(1, -1), neighbors)
    distance_vector = distance_vector[distance_vector != 0]
    numerator = ((1/distance_vector)**n_features).sum()
    core_dist = (numerator / (n_neighbors - 1)) ** (-1/n_features)
    return core_dist


def _mutual_reachability_dist(point_i, point_j, neighbors_i,
                              neighbors_j, dist_function):
    """.
    Computes the mutual reachability distance between points

    Args:
        point_i (np.array): array of dimensions (n_features,)
            point i to compare to point j
        point_j (np.array): array of dimensions (n_features,)
            point i to compare to point i
        neighbors_i (np.ndarray): array of dims (n_neighbors, n_features):
            array of all other points in object class of point i
        neighbors_j (np.ndarray): array of dims (n_neighbors, n_features):
            array of all other points in object class of point j
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point

    Returns: mutual_reachability (float)
        mutual reachability between points i and j

    """
    core_dist_i = _core_dist(point_i, neighbors_i, dist_function)
    core_dist_j = _core_dist(point_j, neighbors_j, dist_function)
    dist = dist_function(point_i, point_j)
    mutual_reachability = np.max([core_dist_i, core_dist_j, dist])
    return mutual_reachability


def _mutual_reach_dist_graph(X, labels, dist_function):
    """
    Computes the mutual reach distance complete graph.
    Graph of all pair-wise mutual reachability distances between points

    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point

    Returns: graph (np.ndarray)
        array of dimensions (n_samples, n_samples)
        Graph of all pair-wise mutual reachability distances between points.

    """
    n_samples = np.shape(X)[0]
    graph = []
    counter = 0
    for row in range(n_samples):
        graph_row = []
        for col in range(n_samples):
            point_i = X[row]
            point_j = X[col]
            class_i = labels[row]
            class_j = labels[col]
            members_i = _get_label_members(X, labels, class_i)
            members_j = _get_label_members(X, labels, class_j)
            dist = _mutual_reachability_dist(point_i, point_j,
                                             members_i, members_j,
                                             dist_function)
            graph_row.append(dist)
        counter += 1
        graph.append(graph_row)
    graph = np.array(graph)
    return graph


def _mutual_reach_dist_MST(dist_tree):
    """
    Computes minimum spanning tree of the mutual reach distance complete graph

    Args:
        dist_tree (np.ndarray): array of dimensions (n_samples, n_samples)
            Graph of all pair-wise mutual reachability distances
            between points.

    Returns: minimum_spanning_tree (np.ndarray)
        array of dimensions (n_samples, n_samples)
        minimum spanning tree of all pair-wise mutual reachability
            distances between points.
    """
    mst = minimum_spanning_tree(dist_tree).toarray()
    return mst + np.transpose(mst)


def _cluster_density_sparseness(MST, labels, cluster):
    """
    Computes the cluster density sparseness, the minimum density
        within a cluster

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest

    Returns: cluster_density_sparseness (float)
        value corresponding to the minimum density within a cluster
    """
    indices = np.where(labels == cluster)[0]
    cluster_MST = MST[indices][:, indices]
    cluster_density_sparseness = np.max(cluster_MST)
    return cluster_density_sparseness


def _cluster_density_separation(MST, labels, cluster_i, cluster_j):
    """
    Computes the density separation between two clusters, the maximum
        density between clusters.

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster_i (int): cluster i of interest
        cluster_j (int): cluster j of interest

    Returns: density_separation (float):
        value corresponding to the maximum density between clusters
    """
    indices_i = np.where(labels == cluster_i)[0]
    indices_j = np.where(labels == cluster_j)[0]
    shortest_paths = csgraph.dijkstra(MST, indices=indices_i)
    relevant_paths = shortest_paths[:, indices_j]
    density_separation = np.min(relevant_paths)
    return density_separation


def _cluster_validity_index(MST, labels, cluster):
    """
    Computes the validity of a cluster (validity of assignmnets)

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest

    Returns: cluster_validity (float)
        value corresponding to the validity of cluster assignments
    """
    min_density_separation = np.inf
    for cluster_j in np.unique(labels):
        if cluster_j != cluster:
            cluster_density_separation = _cluster_density_separation(MST,
                                                                     labels,
                                                                     cluster,
                                                                     cluster_j)
            if cluster_density_separation < min_density_separation:
                min_density_separation = cluster_density_separation
    cluster_density_sparseness = _cluster_density_sparseness(MST,
                                                             labels,
                                                             cluster)
    numerator = min_density_separation - cluster_density_sparseness
    denominator = np.max([min_density_separation, cluster_density_sparseness])
    cluster_validity = numerator / denominator
    return cluster_validity


def _clustering_validity_index(MST, labels):
    """
    Computes the validity of all clustering assignments for a
    clustering algorithm

    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X

    Returns: validity_index (float):
        score in range[-1, 1] indicating validity of clustering assignments
    """
    n_samples = len(labels)
    validity_index = 0
    for label in np.unique(labels):
        fraction = np.sum(labels == label) / float(n_samples)
        cluster_validity = _cluster_validity_index(MST, labels, label)
        validity_index += fraction * cluster_validity
    return validity_index


def _get_label_members(X, labels, cluster):
    """
    Helper function to get samples of a specified cluster.

    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest

    Returns: members (np.ndarray)
        array of dimensions (n_samples, n_features) of samples of the
        specified cluster.
    """
    indices = np.where(labels == cluster)[0]
    members = X[indices]
    return members
