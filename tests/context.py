# See https://docs.python-guide.org/writing/structure/#test-suite
import os
import sys
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(parent_dir))

import audtorch  # noqa: E402, F401
