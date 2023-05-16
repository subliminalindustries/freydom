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

parser.add_argument(
    '-b',
    '--block-size',
    help='fft block size (lower means wider filter frequency bandwidth but more accurately generated temporal waveform)',
    type=int,
    nargs=1,
    default=512,
    action='store'
)

parser.add_argument(
    '-l',
    '--band-start',
    help='filter band start frequency (Hz)',
    type=float,
    nargs=1,
    default=7000.0,
    action='store'
)

parser.add_argument(
    '-r',
    '--band-stop',
    help='filter band stop frequency (Hz)',
    type=float,
    nargs=1,
    default=15000.0,
    action='store'
)

parser.add_argument(
    '-c',
    '--convolve',
    help='convolve output with input (can yield more audible results under some circumstances but noisy)',
    default=False,
    action='store_true'
)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.filename is None:
        parser.print_help()
        exit(0)

    print(args)
    fve = FreyVoiceEnhancer().process(args.filename, block_size=args.block_size, flt_band_start=args.band_start,
                                      flt_band_stop=args.band_stop, convolve=args.convolve)
