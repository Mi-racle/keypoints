import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=ROOT / 'data')
    parser.add_argument('--epochs', default=100)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    print(opt)


if __name__ == '__main__':
    main()
