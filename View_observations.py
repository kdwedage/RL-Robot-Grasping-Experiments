import numpy as np
from matplotlib import pyplot as plt
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--filename', type=str, default='observations.npy', help='Filename to load the observations from.')
args = argparser.parse_args()
data = np.load(args.filename)
plt.imshow(data[0])
plt.show()
print(data.shape)
# plt.savefig('observation.png')