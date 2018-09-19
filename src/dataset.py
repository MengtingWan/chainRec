import numpy as np
import pandas as pd
import gzip
import sys

DATA_DIR = "./data/"

class Dataset(object):
    
    def __init__(self, DATA_NAME, n_stage):
        self.DATA_NAME = DATA_NAME
        
        print("Initializing dataset:", DATA_NAME)
        sys.stdout.flush()
        
        DATA_PATH = DATA_DIR+DATA_NAME+".user_item_map.gz"
        n_user = 0
        n_item = 0
        n_interaction = 0
        try:
            user_item_map = {}
            with gzip.open(DATA_DIR+DATA_NAME+".user_item_map.gz") as fin:
                for l in fin:
                    d = eval(l)
                    uid = int(d['user_id'])
                    if (uid+1) > n_user:
                        n_user = (uid+1)
                    
                    items = np.array(list(d['items'].items()), dtype=int)
                    n_interaction += items.shape[0]
                    
                    max_iid = items[:,0].max()
                    if (max_iid+1) > n_item:
                        n_item = (max_iid+1)
                    user_item_map[uid] = items
        except:
            print("Fail to load", DATA_PATH, ". Please check if the file exists and is in a correct format!")
        
        self.user_item_map = user_item_map
        self.n_user = n_user
        self.n_item = n_item
        self.n_stage = n_stage
        self.n_interaction = n_interaction
        
        print("Successfully initialized!")
        print(n_interaction, "interactions about", n_user, "users,", n_item, "items,", n_stage, "stages are loaded!")
        sys.stdout.flush()
    
    
    def split_train_test(self, method="byUser", seed=None, max_validation_test_samples=None,
                         train_ratio=0.8, vali_ratio=0.1, dump_splits=False):
        
        user_item_map = self.user_item_map
        data_test = []
        if max_validation_test_samples is None:
            max_validation_test_samples = self.n_interaction
        
        print("Spliting data ...")
        sys.stdout.flush()
        np.random.seed(seed)
        if method == "byUser":
            n_vali_samples = 0
            for (uid, items) in user_item_map.items():
                n_items_u = items.shape[0]
                if n_items_u >= 3 and n_vali_samples < max_validation_test_samples:
                    vali_test_index = np.random.choice(n_items_u, size=2, replace=False)
                    data_test.append([uid, items[vali_test_index[0],0], items[vali_test_index[0],1],
                                      items[vali_test_index[1],0], items[vali_test_index[1],1]])
                    n_vali_samples += 1
            data_test = pd.DataFrame(np.array(data_test, dtype=int), 
                                     columns=["user_id", "item_id_vali", "max_stage_vali",
                                              "item_id_test", "max_stage_test"])
            self.data_test = data_test
            print("Successfully splited data in to train/validation/test!")
            sys.stdout.flush()
            if dump_splits:
                TEST_FILE_PATH = DATA_DIR + self.DATA_NAME + ".test.csv"
                data_test.to_csv(TEST_FILE_PATH)
        else:
            print("Fail to split data. Please check if proper spliting parameters are specified.")
            sys.stdout.flush()
            
    
    def sampling_validation(self, dump_samples=False):
        n_item = self.n_item
        user_item_map = self.user_item_map        
        try:
            dict_test = dict(zip(self.data_test['user_id'].values, 
                                 self.data_test[["item_id_vali", "max_stage_vali", 
                                                "item_id_test", "max_stage_test"]].values))
        except:
            print("Fail to load data splits.")
            sys.stdout.flush()

        print("Start sampling validation data ...")
        sys.stdout.flush()
        
        validation_samples = []
        count = 0
        print("current user: ", end="")
        sys.stdout.flush()
        
        for u in dict_test:
            if count % 5000 == 0:
                print(count, end=", ")
                sys.stdout.flush()   
            count += 1
            
            items = user_item_map[u]
            
            item_test = dict_test[u]
            iid_vali = item_test[0]
            max_stage = item_test[1] + 1

            for sid in range(max_stage):      
                items_s = items[items[:,1]>=sid,0]
                p_s = np.ones(n_item)
                p_s[items_s] = 0
                p_s /= np.sum(p_s)
                neg_i = np.random.choice(n_item, p=p_s)
                validation_samples.append([u, iid_vali, sid, neg_i])
        validation_samples = np.array(validation_samples, dtype=int)
        print("done!")
        sys.stdout.flush()    
        
        if dump_samples:
            VALI_FILE_PATH = DATA_DIR + self.DATA_NAME + ".validation_samples.csv"
            pd.DataFrame(validation_samples).to_csv(VALI_FILE_PATH, header=None)
        
        return validation_samples
        
    
    def sampling_training(self, method="edgeOpt_uniform", include_all_pos=True, N_TRAIN=1000000, N_NEG=5, dump_samples=False):
        
        n_item = self.n_item
        n_stage = self.n_stage
        n_interaction = self.n_interaction
        user_item_map = self.user_item_map
        try:
            dict_test = dict(zip(self.data_test['user_id'].values, 
                                 self.data_test[["item_id_vali", "max_stage_vali", 
                                                "item_id_test", "max_stage_test"]].values))
        except:
            print("Fail to load data splits.")
            sys.stdout.flush()
        
        sample_method_set = set(["sliceOpt", "condOpt", "edgeOpt_uniform", "edgeOpt_stage"])
        if method in sample_method_set:
            print("Current sampling method:", method)
        else:
            print("Sampling method is not correctly specified!")
            return None
        
        print("Start sampling training data ...")
        sys.stdout.flush()
        
        if method == "sliceOpt":
            
            training_samples = []
            count = 0
            print("current user: ", end="")
            sys.stdout.flush()       
            for u in user_item_map:
                if count % 5000 == 0:
                    print(count, end=", ")
                    sys.stdout.flush()   
                count += 1
                
                items = user_item_map[u]
                p = np.ones(items.shape[0])
                if u in dict_test:
                    item_test = dict_test[u]
                    p[items[:,0]==item_test[0]] = 0
                    p[items[:,0]==item_test[2]] = 0
                p /= np.sum(p)
                
                n_train_u = sum(p)
                if n_train_u > 0:
                    if include_all_pos:
                        train_data_u = items[p>0, :]
                    else:
                        n_train_u = int(items.shape[0]/n_interaction*N_TRAIN) + 1
                        train_data_u = items[np.random.choice(items.shape[0], size=n_train_u, p=p),:]
                    max_stage = train_data_u[:,1].max() + 1
                    
                    for sid in range(max_stage):
                        pos_i_list = train_data_u[train_data_u[:,1]>=sid, 0]
                        n_pos = len(pos_i_list)
                        if n_pos>0:          
                            tmp = np.append(np.ones((n_pos,1))*u, pos_i_list.reshape(n_pos,1), axis=1)
                            tmp = np.append(tmp, np.ones((n_pos,1))*sid, axis=1)
                            items_s = items[items[:,1]>=sid,0]
                            p_s = np.ones(n_item)
                            p_s[items_s] = 0
                            p_s /= np.sum(p_s)
                            neg_i_list = np.random.choice(n_item, size=n_pos*N_NEG, p=p_s)
                            tmp = np.append(tmp, neg_i_list.reshape(n_pos, N_NEG), axis=1)
                            training_samples += list(tmp)
            training_samples = np.array(training_samples, dtype=int)
            print("done!")
            sys.stdout.flush()

        elif method == "condOpt":

            training_samples = []
            count = 0
            print("current user: ", end="")
            sys.stdout.flush()       
            for u in user_item_map:
                if count % 5000 == 0:
                    print(count, end=", ")
                    sys.stdout.flush()   
                count += 1
                
                items = user_item_map[u]
                p = np.ones(items.shape[0])
                if u in dict_test:
                    item_test = dict_test[u]
                    p[items[:,0]==item_test[0]] = 0
                    p[items[:,0]==item_test[2]] = 0
                p /= np.sum(p)
                    
                n_train_u = sum(p)
                if n_train_u > 0:
                    if include_all_pos:
                        train_data_u = items[p>0, :]
                    else:
                        n_train_u = int(items.shape[0]/n_interaction*N_TRAIN) + 1
                        train_data_u = items[np.random.choice(items.shape[0], size=n_train_u, p=p),:]
                    max_stage = train_data_u[:,1].max() + 1
                    
                    for sid in range(max_stage):
                        pos_i_list = train_data_u[train_data_u[:,1]>=sid, 0]
                        n_pos = len(pos_i_list)
                        if n_pos>0:          
                            tmp = np.append(np.ones((n_pos,1))*u, pos_i_list.reshape(n_pos,1), axis=1)
                            tmp = np.append(tmp, np.ones((n_pos,1))*sid, axis=1)
                            if sid == 0:
                                items_s = items[items[:,1]>=sid,0]
                                p_s = np.ones(n_item)
                                p_s[items_s] = 0
                                p_s /= np.sum(p_s)
                                neg_i_list = np.random.choice(n_item, size=n_pos*N_NEG+1, p=p_s)                            
                            else:
                                items_s_neg = items[items[:,1]==(sid-1),0]
                                if len(items_s_neg) > 0:
                                    neg_i_list = np.random.choice(items_s_neg, size=n_pos*N_NEG+1)
                                else:
                                    neg_i_list = np.random.choice(n_item, size=n_pos*N_NEG)
                            tmp = np.append(tmp, neg_i_list.reshape(n_pos, N_NEG), axis=1)
                            training_samples += list(tmp)
            training_samples = np.array(training_samples, dtype=int)
            print("done!")
            sys.stdout.flush()

            
        elif method == "edgeOpt_uniform":            

            training_samples = []
            count = 0
            print("current user: ", end="")
            sys.stdout.flush()       
            for u in user_item_map:
                if count % 5000 == 0:
                    print(count, end=", ")
                    sys.stdout.flush()   
                count += 1
                
                items = user_item_map[u]
                p = np.ones(items.shape[0])
                if u in dict_test:
                    item_test = dict_test[u]
                    p[items[:,0]==item_test[0]] = 0
                    p[items[:,0]==item_test[2]] = 0
                p /= np.sum(p)
                
                stage_map = -np.ones(n_item)
                stage_map[items[:,0]] = items[:,1]
                    
                n_train_u = sum(p)
                if n_train_u > 0:
                    if include_all_pos:
                        train_data_u = items[p>0, :]
                    else:
                        n_train_u = int(items.shape[0]/n_interaction*N_TRAIN) + 1
                        train_data_u = items[np.random.choice(items.shape[0], size=n_train_u, p=p),:]
                    max_stage = train_data_u[:,1].max() + 1
                    
                    for sid in range(max_stage):
                        pos_i_list = train_data_u[train_data_u[:,1]==sid,0]
                        n_pos = len(pos_i_list)                
                        if n_pos>0:          
                            tmp = np.append(np.ones((n_pos,1))*u, pos_i_list.reshape(n_pos,1), axis=1)
                            tmp = np.append(tmp, np.ones((n_pos,1))*sid, axis=1)  
                            neg_i_list = np.random.choice(n_item, size=n_pos*N_NEG)
                            neg_i_stage = stage_map[neg_i_list]
                            tmp_neg = np.array([neg_i_list, neg_i_stage]).transpose()
                            tmp = np.append(tmp, tmp_neg.reshape(n_pos, N_NEG*2), axis=1)
                            training_samples += list(tmp)
            training_samples = np.array(training_samples, dtype=int)
            print("done!")
            sys.stdout.flush()      
        
                    
        elif method == "edgeOpt_stage":

            training_samples = []
            count = 0
            print("current user: ", end="")
            sys.stdout.flush()       
            for u in user_item_map:
                if count % 5000 == 0:
                    print(count, end=", ")
                    sys.stdout.flush()   
                count += 1
                
                items = user_item_map[u]
                p = np.ones(items.shape[0])
                if u in dict_test:
                    item_test = dict_test[u]
                    p[items[:,0]==item_test[0]] = 0
                    p[items[:,0]==item_test[2]] = 0
                p /= np.sum(p)
                
                stage_map = -np.ones(n_item)
                stage_map[items[:,0]] = items[:,1]
                stage_count = np.zeros(n_stage)
                for sid in range(n_stage):
                    stage_count[sid] = np.sum(items[:,1]>sid)/np.sum(items[:,1]>=sid)
                    
                n_train_u = sum(p)
                if n_train_u > 0:
                    if include_all_pos:
                        train_data_u = items[p>0, :]
                    else:
                        n_train_u = int(items.shape[0]/n_interaction*N_TRAIN) + 1
                        train_data_u = items[np.random.choice(items.shape[0], size=n_train_u, p=p),:]
                    max_stage = train_data_u[:,1].max() + 1
                    
                    for sid in range(max_stage):
                        pos_i_list = train_data_u[train_data_u[:,1]==sid,0]
                        n_pos = len(pos_i_list)                
                        if n_pos>0:          
                            tmp = np.append(np.ones((n_pos,1))*u, pos_i_list.reshape(n_pos,1), axis=1)
                            tmp = np.append(tmp, np.ones((n_pos,1))*sid, axis=1)  
                            p = np.ones(n_item)*items.shape[0]/n_item
                            p[items[:,0]] = stage_count[items[:,1]]
                            p /= np.sum(p)
                            neg_i_list = np.random.choice(n_item, size=n_pos*N_NEG, p=p)
                            neg_i_stage = stage_map[neg_i_list]
                            tmp_neg = np.array([neg_i_list, neg_i_stage]).transpose()
                            tmp = np.append(tmp, tmp_neg.reshape(n_pos, N_NEG*2), axis=1)
                            training_samples += list(tmp)
            training_samples = np.array(training_samples, dtype=int)
            print("done!")
            sys.stdout.flush()
        
        if dump_samples:
            TRAIN_FILE_PATH = DATA_DIR + self.DATA_NAME + "." + method + ".training_samples.csv"
            pd.DataFrame(training_samples).to_csv(TRAIN_FILE_PATH, header=None)
            
        return training_samples
            
        
            
