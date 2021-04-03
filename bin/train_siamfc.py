import os
import sys
sys.path.append(os.getcwd())
from fire import Fire

from siamfc import train

if __name__ == '__main__':
    # Fire(train)
    data_dir = 'F:/ILSVRC2015'
    train(0, data_dir)
