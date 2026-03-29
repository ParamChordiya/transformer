import sys
import os
# Allow `from src.foo import Bar` and `from config import Config` in all tests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
