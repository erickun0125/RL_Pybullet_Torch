# src/main.py
import argparse
from train import train_agent
from test import test_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='실행 모드: train 또는 test')
    args = parser.parse_args()

    if args.mode == 'train':
        train_agent()
    elif args.mode == 'test':
        test_agent()

if __name__ == '__main__':
    main()
