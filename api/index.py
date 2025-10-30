import os
import sys

# Ensure project root is on the import path when running as a Vercel function
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app as app


