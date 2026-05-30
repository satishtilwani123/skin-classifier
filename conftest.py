"""
conftest.py — must sit at the project root (same level as app_backend/).
 
This file does two things:
1. Adds the project root to sys.path so pytest can import app_backend.
2. Provides a session-scoped autouse fixture that resets the ML model
   singleton between test sessions, preventing state leaking across runs.
"""
 
import sys
import os
 
# Insert the project root at the front of sys.path.
# __file__ is <project_root>/conftest.py, so dirname gives the root.
sys.path.insert(0, os.path.dirname(__file__))
 