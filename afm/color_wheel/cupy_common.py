#!/usr/bin/env python

try:
    import cupy as cp
except ImportError:
    pass


def check_cupy_available():

    available = False

    try:
        import cupy as cp
        available = True
    except ImportError:
        pass
  
    return available

def memcopy_to_device(host_pointers):

    for key, value in host_pointers.items(): 
        if type(value) is dict: 
            memcopy_to_host(value) #this handles dictionaries of dictionaries
        elif value is not None:
            host_pointers[key] = cp.asarray(value)


def memcopy_to_host(device_pointers):

    for key, value in device_pointers.items():
        if type(value) is dict: 
            memcopy_to_host(value) #this handles dictionaries of dictionaries
        elif value is not None:
            device_pointers[key] = cp.asnumpy(value)  


    

    
