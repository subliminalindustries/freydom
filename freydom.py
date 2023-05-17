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
    help='file to process'
)

parser.add_argument(
    '-s',
    '--block-size',
    help='fft block size (lower means wider filter frequency bandwidth but a more accurate time-domain filter)',
    type=int,
    nargs=1,
    default=512,
    action='store'
)

parser.add_argument(
    '-b',
    '--band-width',
    help='filter band-widths (Hz) (for example: -b 160-300 600-700)',
    type=str,
    nargs='+',
    default='0-120',
    action='store'
)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.filename is None:
        parser.print_help()
        exit(0)

    if type(args.band_width) == str:
        args.band_width = [args.band_width]

    fve = FreyVoiceEnhancer().process(args.filename,
                                      fft_n=args.block_size,
                                      flt_bws=args.band_width)
