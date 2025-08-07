# File: tests/test_memory_agent.py

import os
import shutil
import tempfile
from quark.agents.memory_agent import MemoryAgent

def test_remember_and_recall():
    tmpdir = tempfile.mkdtemp()
    ma = MemoryAgent(memory_dir=tmpdir)
    # remember two items
    ma.generate("The sky is blue.", operation="store_memory", memory_type="episodic")
    ma.generate("Grass is green.", operation="store_memory", memory_type="episodic")
    # recall something related to sky
    docs = ma.generate("sky", operation="retrieve_memories", max_results=1)
    assert "content" in docs or "memories" in docs
    # cleanup
    shutil.rmtree(tmpdir)

