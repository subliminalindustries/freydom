#! /usr/bin/env python3

import sys
import argparse

from freydom import FreyVoiceEnhancer


parser = argparse.ArgumentParser(
    'freydom',
    usage='freydom.py <file>',
    description='Microwave auditory effect vocal content isolator',
    add_help=True
)

parser.add_argument(
    'file',
    help='File to process'
)

args = parser.parse_args()
if args.file is not None:
    fve = FreyVoiceEnhancer(args.file)
    fve.process()
    exit(0)

parser.print_help()
