""""
JPLE: Joint Protein-Ligand Embedding

This package provide tools for training and predicting protein-nucleic acid
interactions using JPLE. It includes functionality for

- Extracting domains
- Generating protein representations
- Training JPLE
- Performing JPLE protein queries and RNA queries
- Computing PWMs and motifs
"""
import logging

# Configure logging behavior
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Package version
__version__ = '1.0.0'
