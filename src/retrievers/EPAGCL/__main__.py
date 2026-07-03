"""
Entry point for running EPAGCL as a module or script.
This allows both:
  - python -m src.retrievers.EPAGCL.main (from project root)
  - python src/retrievers/EPAGCL/main.py (from project root)
"""
from .main import *

if __name__ == '__main__':
    import torch
    import os
    import numpy as np
    from tqdm import tqdm
    from time import time
    
    args = get_args()
    data = get_dataset(args.dataset_path, args.dataset).to(args.device)
    t0 = time()
    adding_edge_weights, adding_edge, dropping_edge_weights = setup(data, args)
    t1 = time()
    print(f"Data Processing Time: {t1-t0:.2f}s")
    output_path = args.output_path
    
    print(f"dataset:{args.dataset}  device:{args.device}  feature dim:{args.feat_dim}")
    acc = []
    for i in range(args.repeat):
        args.output_path = output_path + '/' + str(i+1)
        ac = train(data, adding_edge_weights, adding_edge, dropping_edge_weights, args)
        acc.append(ac)

    acc = np.array(acc)
    print("max:{:.2f}, min:{:.2f}, mean:{:.2f}, std:{:.2f}".format(acc.max(), acc.min(), acc.mean(), acc.std()))
    f = open(output_path+'/result.txt',"a")
    print(acc, file=f)
    print("mean:{:.2f}, std:{:.2f}".format(acc.mean(), acc.std()), file=f)
    dir = output_path.replace('in-progress', 'completed')
    os.rename(output_path, dir)
    print("result saved to" + dir)
