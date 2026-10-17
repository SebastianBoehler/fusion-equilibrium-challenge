"""Use the unchanged059 experiment with native observed job memory accounting."""
import argparse
from pathlib import Path
from research import full_constraint_support as control
from research.full_constraint_resources_native import Memory


def run(root):
    original=control.Memory
    control.Memory=Memory
    try:
        control.run(root)
    finally:
        control.Memory=original


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
