#! /usr/bin/env python3

import glob, os, argparse

from freydom import FreyVoiceEnhancer


parser = argparse.ArgumentParser(
    'freydom',
    usage='freydom.py <file>',
    description='Microwave auditory effect vocal content isolator',
    add_help=True
)

parser.add_argument(
    'filename',
    help='File to process'
)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.filename is None:
        parser.print_help()
        exit(0)

    fve = FreyVoiceEnhancer().process(args.filename)
