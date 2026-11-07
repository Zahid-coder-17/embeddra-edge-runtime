import os, sys

# Allow both absolute package and relative local execution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
edge_dir = os.path.join(parent_dir, "embeddra-edge")
if edge_dir not in sys.path:
    sys.path.insert(0, edge_dir)

