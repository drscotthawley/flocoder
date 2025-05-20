# Coding Style Guide

*"Do as I say, not as I do..."*

This repo is intended to facilitate *teaching* and open-ended *experimentation*.  (It is not intended for optimal performance in deployment.) Thus the code should be easy to follow, easy to take in "at a glance", easy to trace the execution, and easy to see where to modify something to add new behavior, *without having to consult lengthy documentation on Class methods and/or scheduling, or diving through multiple levels of inheritance and/or imports*. 

## 1. Vertical compactness

Wherever possible, code should be designed to take up as little vertical space as possible, to facilitate taking in the flow of execution "at a glance." 

No lengthy docstrings, use inline "docments" (https://fastcore.fast.ai/docments.html) for args & kwargs. Example use of inline documents:  

```python
def mypow(x,            # here's a description of x
          b:int,        # type hints are optional but welcome!
          dummy=None):  # hey it's a kwarg
	"""Raises x to the power of b"""
	return x**b
  
```



 When calling functions, args & kwargs run sideways rather than vertically. 

Comments that only refer to a single line of code should go to the right rather than above it.

Single-line if-then blocks are perfectly fine

"Exceptions": 

- It's ok to define variables that only get used once, if doing so facilitates the readability of the code.  



## 2. Linear Execution

...rather than hierarchy of (sub)-Classes.  We do not have a Trainer class, or an Evaluator class, etc.  We have a big long list of exposed code so that you know where you are and you don't have to dive through multiple levels to figure out where something occurs or where to change something. 

This guideline may seem to be at odds with the vertical compactness guideline -- "Surely if we made the code more modular, it would take up less vertical space."  Actually it's the other way around: because we want linear execution, then having it vertically compact allows for better comprehension.

Classes are appropriate for custom Datasets, and neural network models, and other cases where you really want to have a dedicated namespace for something akin to global variables, and/or you want certain methods to be attached to certain objects.  We're not utterly opposed to classes or even inheritance, but some libraries take this to such an extreme with nested this-and-thats that unless you know "where everything is", you will have a hard time tracing the flow and/or figuring out where to add your new idea.



## 3. Flexibility & Adaptability

(This one is really *not* practiced in the current code.)  Functions and methods should ideally acquire their needed parameters from the variables passed in, e.g. for things like shapes and devices.  These should not be hard-coded...but they still are in some places, so we should fix that.  

When we start to have way too many args & kwargs, its's acceptable to switch over to passing a dict in. 



## Note

None of these guidelines are followed perfectly throughout the code.  Nevertheless, these are the "soft constraints".
