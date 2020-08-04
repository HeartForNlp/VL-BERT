import os
import numpy as np

captions = []
urls = []

with open('Train_GCC-training.tsv') as fp:
    for cnt, line in enumerate(fp):
        s = line.split('\t')
        captions.append(s[0].split(' '))
        urls.append(s[1][:-1])
np.random.seed(42)
urls = list(np.random.choice(urls, int(0.01*len(urls)), replace=False))
        
with open('train4download.txt', 'w') as fp:
    for cnt, url in enumerate(urls):
        fp.write("../train_image/{:08d}.jpg\t\"{}\"\n".format(cnt, url))

if not os.path.exists('../train_image'):
    os.makedirs('../train_image')
