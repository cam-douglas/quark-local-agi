# File: tests/test_memory_agent.py

import os
import shutil
import tempfile
from memory_agent import MemoryAgent

def test_remember_and_recall():
    tmpdir = tempfile.mkdtemp()
    ma = MemoryAgent(db_path=tmpdir)
    # remember two items
    ma.remember("k1", "The sky is blue.")
    ma.remember("k2", "Grass is green.")
    # recall something related to sky
    docs = ma.recall("sky", top_k=1)
    assert "blue" in docs[0]
    # cleanup
    shutil.rmtree(tmpdir)

