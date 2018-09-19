import numpy as np
import pandas as pd
import gzip
import sys
from collections import Counter
import os

DATA_DIR = "../data/"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
    
def process_yoochoose():
    print("loading yoochoose data ... ")   
    sys.stdout.flush()
    
    data_buy = pd.read_csv(DATA_DIR+"yoochoose-buys.dat", header=None)
    data_buy.columns = ['session_id', 'ts', 'item_id', 'category', 'Qty']
    data = dict([(k, dict(zip(v.values, np.ones(len(v)).astype(int)))) for (k, v) in data_buy.groupby(['session_id'])['item_id']])
    with open(DATA_DIR+"yoochoose-clicks.dat") as fin:
        for l in fin:
            sid, _, iid, _ = l.strip().split(",")
            sid = int(sid)
            iid = int(iid)
            if sid in data:
                dout = data[sid]
                if iid not in dout:
                    dout[iid] = 0
                data[sid] = dout
    item_count = np.array(list(Counter([i for u in data for i in data[u]]).items()))
    
    user_map = {}
    item_set = set(list(item_count[item_count[:,1].astype(float)>=5,0]))
    item_map = {}
    print("preprocessing and dumping data ... ")
    sys.stdout.flush()
    with gzip.open(DATA_DIR+"yoochoose.user_item_map.gz", "w") as fdata:
        for uid_str in data:
            d0 = data[uid_str]
            d = {}
            for di in d0:
                if di in item_set:
                    iid = len(item_map)
                    if di in item_map:
                        iid = item_map[di]
                    else:
                        item_map[di] = iid
                    d[iid] = d0[di]
            if len(d)>0:
                dout = {}
                uid = len(user_map)
                if uid_str in user_map:
                    uid = user_map[uid_str]
                else:
                    user_map[uid_str] = uid
                dout['user_id'] = uid
                dout['items'] = d
                fdata.write((str(dout)+"\n").encode("utf-8"))
    np.savetxt(DATA_DIR+"yoochoose_user_names.csv", np.array(list(user_map.items())), fmt="%s", delimiter=",")
    np.savetxt(DATA_DIR+"yoochoose_item_names.csv", np.array(list(item_map.items())), fmt="%s", delimiter=",")
    print("done!")
    sys.stdout.flush()