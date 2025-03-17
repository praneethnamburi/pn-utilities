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
import weakref
from copy import deepcopy
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd

if os.name == 'nt':
    import multiprocess

import blinker
import numpy as np

try:
    BLENDER_MODE = True
    import bpy
except ImportError:
    BLENDER_MODE = False

from pyfilemanager import FileManager, find, get_file_sizes


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


## Event handlers and broadcasting using blinker's signal
class Handler:
    """
    Event handlers based on blinker's signal.
    Currently, handlers can be defined on:
    1) Class functions - act on all members of a class
    2) Bound methods - act on a specific class member
    3) Class property - act on all members of a class when a property is set
    4) Object property - act on a specific class member when its property is set
    A receiver function can be attached either before, or after for each of these categories.
    Therefore, there are 8 types of handlers in total.
    thing = (class, object)
    attr = (function, property)
    mode = (pre, post)
    """
    def __init__(self, thing, attr, mode='post', sig=None):
        assert isinstance(attr, str)
        assert hasattr(thing, attr)
        assert mode in ('pre', 'post')
        self._thing = weakref.ref(thing)
        self.attr = attr
        self.mode = mode
        if sig is None:
            self.signal = blinker.base.signal
        else: # providing signal from a specific namespace will leave blinker's default namespace free for other apps
            assert isinstance(sig.__self__, blinker.base.Namespace)
            self.signal = sig
        assert self.attr_cat in ('property', 'function')
        if self.attr_cat == 'property':
            assert getattr(self.thing_class, self.attr).fset is not None
    
    thing = property(lambda s: s._thing())
    thing_is_class = property(lambda s: inspect.isclass(s.thing))
    thing_class = property(lambda s: s.thing if s.thing_is_class else type(s.thing))
    attr_cat = property(lambda s: type(getattr(s.thing_class, s.attr)).__name__)
    mod_name = property(lambda s: s.thing.__module__ if s.thing_is_class else type(s.thing).__module__)
    cls_id = property(lambda s: s.mode + '-' + s.mod_name + '-' + s.thing_class.__name__ + '-' + s.attr_name)
    @property
    def attr_name(self):
        """Name of the attribute. If it is a property, it must have a setter to support a handler."""
        if self.attr_cat == 'function':
            return self.attr
        return self.attr + '.fset'
    @property
    def instance_name(self):
        """Name of the instance"""
        if self.thing_is_class:
            return ''
        return self.thing.name if hasattr(self.thing, 'name') else hex(id(self.thing))
    @property
    def id(self):
        """This is the broadcasted signal."""
        if self.thing_is_class:
            return self.cls_id
        return self.cls_id + '(' + self.instance_name + ')'

    def id2dict(self):
        """Handler ID as a dictionary"""
        return handler_id2dict(self.id)

    def broadcast(self):
        """Tweak thing's attr to broadcast a signal either before or after execution."""
        if self.attr_cat == 'function':
            setattr(self.thing, self.attr, self._broadcast_function())
        if self.attr_cat == 'property': # only the class property can broadcast!            
            # Remember that either all instances broadcast a property, or none of them do.
            # The strategy for object specific handlers is to filter at the receiver.
            setattr(self.thing_class, self.attr, self._broadcast_property())

    def add_receiver(self, receiver_func):
        """
        Add a receiver function to the handler.
        A receiver function should have the same signature as defining a function in a class:
        def receiver_fun(self):
            pass
        """
        assert type(receiver_func).__name__ in ('function', 'method')
        r_desc = self.receiver_descriptor(receiver_func)
        if r_desc not in [r for r in self.receivers if r[1] != '<lambda>']:
            self.signal(self.id).connect(receiver_func)
        else:
            print('Receiver with description '+str(r_desc)+' already connected. No action taken.')
    
    def get_receivers(self):
        """Return the receivers (weakref list)"""
        return self.signal(self.id).receivers

    def delete_receivers(self):
        """Delete all receivers for a signal."""
        self.signal(self.id).receivers = {}

    @property #**
    def channels(self):
        """Broadcasting channels (if any)"""
        if self.attr_cat == 'function':
            func = getattr(self.thing, self.attr)
            if hasattr(func, '__broadcast__'):
                return func.__broadcast__
            return None
        if self.attr_cat == 'property':
            p = getattr(self.thing_class, self.attr)
            if hasattr(p.fset, '__broadcast__'):
                return p.fset.__broadcast__
            return None

    @property #**
    def receivers(self):
        """
        Descriptions for current receiver functions.
        ('function'/'method(obj_id)', __qualname__, __module__)
        """
        return [self.receiver_descriptor(r()) for r in list(self.get_receivers().values())]
    
    def __eq__(self, other):
        return self.channels == other.channels and self.receivers == other.receivers
    
    def __str__(self):
        return object.__repr__(self)

    def __repr__(self):
        return self.__module__ + "." + self.__class__.__name__ + ': ' + self.id + '\nChannels: ' + str(self.channels) + '\nReceivers: ' + str(self.receivers)

    @staticmethod
    def receiver_descriptor(r):
        """Tuple description of a signal's receiver function"""
        f_type = type(r).__name__
        if f_type == 'method':
            bound_obj = r.__self__
            bound_obj_id = bound_obj.name if hasattr(bound_obj, 'name') else hex(id(bound_obj))
            return (f_type+'('+ bound_obj_id +')', r.__qualname__, r.__module__)
        return (f_type, r.__qualname__, r.__module__)

    def _broadcast_function(self):
        """
        modifies self.thing's attribute to broadcast
        """
        func = getattr(self.thing, self.attr)
        func_type = type(func).__name__
        signal_name = self.id

        if hasattr(func, '__broadcast__'): # already broadcasting
            assert func.__broadcast__ == signal_name
            return func

        if func_type == 'method': # 'unbounded'
            meth = func
            func = getattr(meth.__self__.__class__, meth.__name__)

        def _new_func_pre(s, *args, **kwargs):
            if bool(self.signal(signal_name).receivers):
                self.signal(signal_name).send(s) # signal is sent BEFORE the object is modified
            f_out = func(s, *args, **kwargs)
            return f_out
        def _new_func_post(s, *args, **kwargs):
            f_out = func(s, *args, **kwargs)
            if bool(self.signal(signal_name).receivers):
                self.signal(signal_name).send(s) # signal is sent AFTER the object is modified
            return f_out

        _new_func = _new_func_pre if self.mode == 'pre' else _new_func_post
        _new_func.__name__ = func.__name__
        _new_func.__qualname__ = func.__qualname__
        _new_func.__module__ = func.__module__
        _new_func.__broadcast__ = signal_name

        if func_type == 'method': # bind the function to the object
            return _new_func.__get__(meth.__self__)
        
        return _new_func

    def _broadcast_property(self):
        """
        Creates a new property with a modified setter.
        Adds a broadcasting signal to the setter of property p.
        """
        p = getattr(self.thing_class, self.attr)
        signal_name = self.cls_id
        assert isinstance(p, property)

        if hasattr(p.fset, '__broadcast__'):
            if signal_name in p.fset.__broadcast__:
                return p # no need to modify the property

        def _new_fset_pre(x, s): # x is the object whose property is being modified (self)
            # broadcast signal for all members
            if bool(self.signal(signal_name).receivers):
                self.signal(signal_name).send(x)
            # member-specific broadcast
            instance_name = x.name if hasattr(x, 'name') else hex(id(x))
            new_signal_name = signal_name+'('+instance_name+')'
            if bool(self.signal(new_signal_name).receivers):
                self.signal(new_signal_name).send(x) # signal is sent AFTER the object is modified
            f_out = p.fset(x, s)
            return f_out
        def _new_fset_post(x, s): # x is the object whose property is being modified (self)
            f_out = p.fset(x, s)
            # broadcast signal for all members
            if bool(self.signal(signal_name).receivers):
                self.signal(signal_name).send(x)
            # member-specific broadcast
            instance_name = x.name if hasattr(x, 'name') else hex(id(x))
            new_signal_name = signal_name+'('+instance_name+')'
            if bool(self.signal(new_signal_name).receivers):
                self.signal(new_signal_name).send(x) # signal is sent AFTER the object is modified
            return f_out

        _new_fset = _new_fset_pre if self.mode == 'pre' else _new_fset_post
        _new_fset.__name__ = p.fset.__name__
        _new_fset.__qualname__ = p.fset.__qualname__
        _new_fset.__module__ = p.fset.__module__
        if hasattr(p.fset, '__broadcast__'):
            _new_fset.__broadcast__ = p.fset.__broadcast__
        else:
            _new_fset.__broadcast__ = []
        _new_fset.__broadcast__ += [signal_name] # this is the signal name for the class
        return property(p.fget, _new_fset)

def handler_id2dict(k):
    """
    Turn a handler ID into meaningful parts
    A handler id is a string that has the following construction:
    mode-module-class-attribute(instance)
    """
    k_dict = {}
    stg1 = k.split('(')
    k_dict['instance'] = stg1[-1].rstrip(')') if len(stg1) == 2 else ''
    stg2 = stg1[0].split('-')
    assert len(stg2) == 4
    k_dict['mode'], k_dict['module'], k_dict['class'], stg3 = stg2
    k_dict['attr'] = stg3.replace('.fset', '')
    return k_dict

def add_handler(thing, attr, receiver_func, mode='post', sig=None):
    """
    One-liner access to setting up a broadcaster and receiver.

    Example:
        s1 = new.sphere('sph1')
        # s1.frame is a property, and fire fun whenever s1.frame is set
        add_handler(s1, 'frame', fun, mode='pre') 
        # Fire fun when the frame attribute of any instance of core.Object is set
        add_handler(core.Object, 'frame', fun, mode='post')
        # s1.translate is a method, and fire fun whenever s1.translate is invoked!
        add_handler(s1, 'translate', core.Object.show_frame, mode='post')
    """
    h = Handler(thing, attr, mode, sig)
    h.broadcast()
    h.add_receiver(receiver_func)
    return h

# BroadcastProperties is useful for modifying classes when defining them
class BroadcastProperties:
    """
    Enables properties in a class to have event handlers. This
    manipulation 'replaces' a property in a class with a new property
    object.

    Takes a class, and makes chosen properties setter emit a signal on
    every change. Use it as a decorator on classes to broadcast some/all
    property changes. Receiver receives the object after it is changed.

    Example: see tests.test_broadcasting2()

    Usage: (Don't chain with the same property. Chaining below is OK)
        @pn.BroadcastProperties('loc', mode='pre')
        @pn.BroadcastProperties('frame', mode='post')
        class Object(Thing):
            frame = property(...)
            loc = property(...)
    """
    def __init__(self, p_names='ALL', mode='post'):
        assert isinstance(p_names, (str, list, tuple))
        assert mode in ('pre', 'post')
        self.p_names = p_names
        self.mode = mode
    def __call__(self, src_class):
        if isinstance(self.p_names, str) and self.p_names == 'ALL':
            src_properties = {p_name : p for p_name, p in src_class.__dict__.items() if isinstance(p, property)}
        else:
            src_properties = {p_name : p for p_name, p in src_class.__dict__.items() if isinstance(p, property) and p_name in self.p_names}
        for p_name, p in src_properties.items():
            if p.fset is not None:
                h = Handler(src_class, p_name, self.mode)
                h.broadcast()
        return src_class


## File system
def locate_command(thingToFind, requireStr=None, verbose=True):
    """
    Locate an executable on your computer.

    :param thingToFind: string name of the executable (e.g. python)
    :param requireStr: require path to thingToFind to have a certain string
    :returns: Full path (like realpath) to thingToFind if it exists
              Empty string if thing does not exist
    """
    if sys.platform == 'linux' or sys.platform == 'darwin':
        queryCmd = 'which'
    elif sys.platform == 'win32':
        queryCmd = 'where'
    proc = subprocess.Popen(queryCmd+' '+thingToFind, stdout=subprocess.PIPE, shell=True)
    thingPath = proc.communicate()[0].decode('utf-8').rstrip('\n').rstrip('\r')
    if not thingPath:
        print('Terminal cannot find ', thingToFind)
        return ''

    if verbose:
        print('Terminal found: ', thingPath)
    if requireStr is not None:
        if requireStr not in thingPath:
            print('Path to ' + thingToFind + ' does not have ' + requireStr + ' in it!')
            return ''
    return thingPath

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

def ospath(thingToFind, errContent=None):
    """
    Find file or directory.
    
    :param thingToFind: string input to os path
    :param errContent=None: what to show when not found
    :returns: Full path to thingToFind if it exists.
              Empty string if thingToFind does not exist.
    """
    if errContent is None:
        errContent = thingToFind
    if os.path.exists(thingToFind):
        print('Found: ', os.path.realpath(thingToFind))
        return os.path.realpath(thingToFind)
    print('Did not find ', errContent)
    return ''

def change_image_dpi(files, dpi:int=300, return_format:str='tif'):
    """
    Change the dpi of a set of images, example - for publication

    Usage - 
        file_list = find('*.tif', r'C:\Dropbox (MIT)\Manuscripts\20230401 Elastic energy EDM\Premiere pro')
        change_image_dpi(file_list)
    """
    from PIL import Image

    if isinstance(files, str):
        if os.path.isdir(files):
            file_list = find('*.tif', path=files)
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
        filename = find(filename)[0]
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

def module_members(mod, includeSubModules=True):
    """Return members of a module."""
    members = {}
    for name, data in inspect.getmembers(mod):
        if name.startswith('__') or (inspect.ismodule(data) and not includeSubModules):
            continue
        members[name] = str(type(inspect.unwrap(data))).split("'")[1]
    return members

def properties(obj):
    """
    For an instance obj of any class, use pn.properties(obj) for a summary of properties.
    Especially useful in the blender console.
    """
    #pylint:disable=expression-not-assigned
    [print((k, type(getattr(obj, k)), np.shape(getattr(obj, k)))) for k in dir(obj) if '_' not in k and 'method' not in k]


## input management
def clean_kwargs(kwargs, kwargs_def, kwargs_alias=None):
    """
    Clean keyword arguments based on default values and aliasing.

    :param kwargs: (dict) input kwargs that require cleaning.
    :param kwargs_def: (dict) should have all the possible keyword arguments.
    :param kwargs_alias: (dict) lists all possible aliases for each keyword.
        {kw1: [kw1, kw1_alias1, ..., kw1_aliasn], ...}
        kw1 is used inside the function, but kw1=val, kw1_alias1=val, ..., kw1_aliasn are all valid

    Returns: 
        (dict) keyword arguments after cleaning. Ensures all keywords in kwargs_def are present, and have the names used in the function.
        (dict) remaining keyword arguments
    """
    if not kwargs_alias:
        kwargs_alias = {key : [key] for key in kwargs_def.keys()}
    kwargs_fun = deepcopy(kwargs_def)
    kwargs_out = deepcopy(kwargs)
    for k in kwargs_fun:
        for ka in kwargs_alias[k]:
            if ka in kwargs:
                kwargs_fun[k] = kwargs_out.pop(ka)

    return kwargs_fun, kwargs_out


## Code development
def reload(constraint='Workspace'):
    """
    Reloads all modules in sys with a specified constraint.
    :param constraint: (str) name to be present within the module's path for reload
    Returns:
        names of all the modules that were identified for reload.
    """
    all_mod = [mod for key, mod in sys.modules.items() if constraint in str(mod)]
    reloaded_mod = []
    for mod in all_mod:
        try:
            importlib.reload(mod)
            reloaded_mod.append(mod.__name__)
        except: # pylint: disable=bare-except 
            #Using a specific exception creates a problem when developing with runpy (Blender development plugin workflow)
            if '<run_path>' not in  mod.__name__:
                print('Could not reload ' + mod.__name__)
    return reloaded_mod

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

if os.name == 'nt':
    class Spawn:
        def __init__(self, func):
            self.func = func
        def __call__(self, *args, **kwargs):
            self._q = multiprocess.Queue()
            self._proc = multiprocess.Process(target=self.func, args=(self._q, *args), kwargs=kwargs)
            self._proc.start()
            return self
        def __neg__(self):
            self._q.put('done')
            self._proc.terminate()
        def send(self, msg):
            self._q.put(msg)


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
                all_proc.append(subprocess.Popen(cmds[cmd_count], shell=True, creationflags=0x00000008))
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

def find_nearest(x, y):
    """
    Find the nearest x-values for every value in y.
    x and y are expected to be lists of floats.
    Returns:
        List with the same number of values as y, but each value is the closest value in x.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return [x[np.argmin(np.abs(x - yi))] for yi in y]


## Simple operations
def split_filename(fname:str) -> tuple:
    name, ext = os.path.splitext(os.path.basename(fname))
    path = os.path.dirname(fname)
    return path, name, ext

def scale_data(d:np.ndarray, d_lim:tuple=None, clip:bool=True) -> np.ndarray: # scale the input between 0 and 1
    """Scale data in a numpy array such that the entries in d_lim scale to (0,1)"""
    if d_lim is None:
        d_lim = (np.min(d), np.max(d))
    do = d_lim[0]
    dw = d_lim[1] - d_lim[0]
    if clip:
        d[d < d_lim[0]] = np.nan
        d[d > d_lim[1]] = np.nan
    return (d - do)/dw

def find_nearest_idx_val(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    val = array[idx]
    return idx, val

def find_nearest_idx(array, value):
    """Index of the nearest value in the array"""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def find_nearest_val(array, value):
    return array[find_nearest_idx(array, value)]

class Mapping:
    """Create a dictionary map between any two columns of a dataframe"""
    def __init__(self, df:pd.DataFrame):
        self.df = df
    
    def __call__(self, left_col_name:str, right_col_name:str, row_selector=None) -> dict:
        if row_selector is None:
            row_selector = lambda k,v: True
        ret = pd.Series(self.df[right_col_name].values,index=self.df[left_col_name]).to_dict()
        return {k:v for k,v in ret.items() if row_selector(k,v)}
    
## Statistics
def p_str(p_val, sep=''):
    """
    Convert p-value to a string to be used in plots
    sep is a separator between stars and p_value
    e.g. p_str(0.02) -> '*p=0.02'
    """
    if np.isnan(p_val):
        return ''
    def p_star(p_val):
        return [ps for ps, sel in zip(('n.s. ', '*', '**', '***'), (0.05<=p_val<=1., 0.01<=p_val<0.05, 0.001<=p_val<0.01, p_val<0.001)) if sel][0]
    return f'{p_star(p_val)}{sep}p={p_val:.2g}'

## matplotlib-specific stuff
def ticks_from_times(times, tick_lim):
    """Generate x, y arrays to supply to plt.plot function to plot a set of x-values (times) as ticks."""
    def nan_pad_x(inp):
            return [item for x in inp for item in (x, x, np.nan)]
    def nan_pad_y(ylim, n):
        return [item for y1, y2 in [ylim]*n for item in (y1, y2, np.nan)]
    return nan_pad_x(times), nan_pad_y(tick_lim, len(times))

if not BLENDER_MODE:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy import stats as sstats
    mpl.rcParams['lines.linewidth'] = 0.75

    PLOT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
    def format_legend(ax):
        """Set the legend labels using 'label' field in plot."""
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def scatter_bar(x, y, ax=None, yl=None, xcat=None, **kwargs):
        """Make a scatter plot with bars. 
        DO NOT USE THIS DIRECTLY. USE scatter_bar_cols.
        """
        scatter_color = kwargs.get('scatter_color', 'dodgerblue')
        bar_color = kwargs.get('bar_color', 'darkblue')
        bar_alpha = kwargs.get('bar_alpha', 1)
        scatter_alpha = kwargs.get('scatter_alpha', 1)
        bar_width = kwargs.get('bar_width', 0.05)
        errorbar_width = kwargs.get('errorbar_width', bar_width*0.4)
        paired = kwargs.get('paired', False)
        if xcat is None:
            x = np.asarray(x)
            y = np.asarray(y)
            xcat = np.unique(x[~np.isnan(x)])
        else:
            x, y = list(zip(*[(this_x, this_y) for this_x, this_y in zip(x, y) if this_x in xcat]))
            x = np.asarray(x)
            y = np.asarray(y)

        if ax is None:
            _, ax = plt.subplots()
            plt_show = True
        else:
            plt_show = False

        ax.scatter(x, y, s=80, color=scatter_color, facecolors='none', alpha=scatter_alpha)
        if isinstance(bar_color, list) and paired:
            group_len = len(y) // len(bar_color)
            x_group = x.reshape(group_len, -1, order='F')
            y_group = y.reshape(group_len, -1, order='F')

            for x_g, y_g in zip(x_group, y_group):
                ax.plot(x_g, y_g, 'k')

        if yl is None:
            yl = ax.get_ylim()
        else:
            ax.set_ylim(yl)
        bar_start = kwargs.get('bar_start', yl[0])
        x_mul = np.diff(ax.get_xlim())
        for cnt, this_x in enumerate(xcat):
            if isinstance(bar_color, list) and len(bar_color) == len(xcat):
                this_color_bar = bar_color[cnt]
            else:
                this_color_bar = bar_color
            this_y = y[x==this_x]
            mu = np.nanmean(this_y)
            n = np.sum(~np.isnan(this_y))
            sem = np.nanstd(this_y)/np.sqrt(n)
            ax.plot([this_x-bar_width*x_mul, this_x-bar_width*x_mul, this_x+bar_width*x_mul, this_x+bar_width*x_mul], [bar_start, mu, mu, bar_start], color=this_color_bar, linewidth=1.2, alpha=bar_alpha)
            ax.plot([this_x, this_x], [mu, mu+sem], color=this_color_bar, linewidth=1.2, alpha=bar_alpha)
            ax.plot([this_x-errorbar_width*x_mul, this_x+errorbar_width*x_mul], [mu+sem, mu+sem], color=this_color_bar, linewidth=1.2, alpha=bar_alpha)
        ax.set_xticks(xcat, xcat)
        if plt_show:
            plt.show(block=False)
        return ax

    def scatter_bar_cols(col_data, col_names, ax=None, yl=None, **kwargs):
        """
        Plot column data  (like in prism).
        col_data  - (list of lists of floats) nan values are excluded before plotting and doing statistics
        col_names - (tuple of strings) column names
        kwargs - 
            size        - (2-tuple) figure size in inches, default (3.75, 6)
            x_buffer    - (float) gap to the left and right of the bars, default 0.5
            show_stats  - (bool) show p-value in the bar graph, only relevant if there are two bars, default True
            show_n      - (bool) show n-values at the bottom of the bars, default True
        """
        assert len(col_data) == len(col_names)
        for col_count, this_col_data in enumerate(col_data):
            this_col_data = np.asarray(this_col_data)
            this_col_data = this_col_data[~np.isnan(this_col_data)]
            col_data[col_count] = list(this_col_data)
        color_palette = kwargs.get('color_palette', PLOT_COLORS)

        # stats arguments
        alternative = kwargs.pop('alternative', 'two-sided')
        equal_var = kwargs.pop('equal_var', True)
        permutations = kwargs.pop('permutations', None)
        paired = kwargs.get('paired', False)

        x = []
        y = []
        scatter_color = []
        bar_color = []
        for col_num in range(len(col_names)):
            y += col_data[col_num]
            x += [col_num]*len(col_data[col_num])
            scatter_color += [color_palette[col_num%len(color_palette)]]*len(col_data[col_num])
            bar_color.append(color_palette[col_num%len(color_palette)])
        xcat = np.unique(np.array(x)[~np.isnan(x)])

        if ax is None:
            f, ax = plt.subplots()
            plt_show = True
            f.set_size_inches(*kwargs.get('size', (3.75, 6)))
        else:
            f = ax.figure
            plt_show = False
        
        x_buffer = (xcat[-1] - xcat[0])*kwargs.get('x_buffer', 0.5)
        ax.set_xlim(xcat[0]-x_buffer, xcat[-1]+x_buffer)
        ax = scatter_bar(x, y, xcat=xcat, ax=ax, yl=yl, scatter_color=scatter_color, bar_color=bar_color, **kwargs)
        if kwargs.get('show_n', True):
            n_str = lambda col_data : f'(n={len(col_data)})'
        else:
            n_str = lambda _ : ''
        ax.set_xticks(xcat, [x+'\n'+n_str(col_data) for x, col_data in zip(col_names, col_data)])

        if len(col_names) == 2 and kwargs.get('show_stats', True):
            if paired:
                st = sstats.ttest_rel(col_data[0], col_data[1], alternative=alternative)
            else:
                st = sstats.ttest_ind(col_data[0], col_data[1], alternative=alternative, equal_var=equal_var, permutations=permutations)
            p_val = st.pvalue
            print(st)
            ax.text(0.5, 0.94, p_str(p_val), transform=ax.transAxes, ha='center', va='bottom')
            yl = ax.get_ylim()
            yc = yl[0] + np.diff(yl)*0.93
            ax.plot(xcat, [yc]*2, color='black', linewidth=1.5)
        if plt_show:
            plt.show(block=False)
        return ax


### ------- FROM STACK OVERFLOW
# Sadly, Python fails to provide the following magic number for us.
ERROR_INVALID_NAME = 123
'''
Windows-specific error code indicating an invalid pathname.

See Also
----------
https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-
    Official listing of all such codes.
'''

def is_pathname_valid(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?

def is_path_creatable(pathname: str) -> bool:
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)

def is_path_exists_or_creatable(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS _and_
    either currently exists or is hypothetically creatable; `False` otherwise.

    This function is guaranteed to _never_ raise exceptions.
    '''
    try:
        # To prevent "os" module calls from raising undesirable exceptions on
        # invalid pathnames, is_pathname_valid() is explicitly called first.
        return is_pathname_valid(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname))
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught here.
    except OSError:
        return False

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
                fm = FileManager(fname)
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

class List(list):
    def next(self, val):
        """Next element in the list closest to val."""
        return min([x for x in self if x > val], default=max(self))
    
    def previous(self, val):
        """Previous element in the list closest to val."""
        return max([x for x in self if x < val], default=min(self))


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
