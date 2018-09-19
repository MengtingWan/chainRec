import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import os

DATA_DIR = "./data/"
MODEL_DIR = "./models/"
OUTPUT_DIR = "./results/"

for DIR in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    if not os.path.exists(DIR):
        os.makedirs(DIR)

class chainRec(object):
    
    def __init__(self, n_user, n_item, n_stage, DATA_NAME):
        self.DATA_NAME = DATA_NAME
        self.n_user = n_user
        self.n_item = n_item
        self.n_stage = n_stage

    def config_global(self, MODEL_CLASS, HIDDEN_DIM, LAMBDA, LEARNING_RATE, BATCH_SIZE, target_stage_id):        
        self.HIDDEN_DIM = HIDDEN_DIM
        self.LAMBDA = LAMBDA
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.MODEL_CLASS = MODEL_CLASS
        self.target_stage_id = target_stage_id
        
        DATA_NAME = self.DATA_NAME
        self.MODEL_NAME = DATA_NAME+"."+MODEL_CLASS+".dim."+str(HIDDEN_DIM)+".lambda."+str(LAMBDA)
        
        
    def load_samples(self, training_samples, validation_samples):
        target_stage_id = self.target_stage_id
        self.training_samples = training_samples
        self.validation_samples = validation_samples[validation_samples[:,2]==target_stage_id, :]
        
    def load_samples_from_files(self, method="edgeOpt_uniform"):
        target_stage_id = self.target_stage_id
        
        TRAIN_FILE_PATH = DATA_DIR + self.DATA_NAME + "." + method + ".training_samples.csv"
        training_samples = pd.read_csv(TRAIN_FILE_PATH, header=None).values
        VALI_FILE_PATH = DATA_DIR + self.DATA_NAME + "." + method + ".validation_samples.csv"
        validation_samples = pd.read_csv(VALI_FILE_PATH, header=None).values
        
        self.training_samples = training_samples
        self.validation_samples = validation_samples[validation_samples[:,2]==target_stage_id, :]
        
    def next_training_batch_edgeOpt(self, BATCH_SIZE, N_MAX=1000000):
        n_neg = (self.training_samples.shape[1] - 3)//2
        N_BATCH = N_MAX//BATCH_SIZE
        index_selected = np.random.permutation(self.training_samples.shape[0])[:N_MAX]
        for i in range(0, N_BATCH*BATCH_SIZE, BATCH_SIZE):
            current_index = index_selected[i:(i+BATCH_SIZE)]
            neg_index = np.random.choice(n_neg, size=current_index.shape[0])
            xu1 = self.training_samples[current_index, 0]
            xi1 = self.training_samples[current_index, 1]
            li1 = self.training_samples[current_index, 2]
            xj1 = self.training_samples[current_index, neg_index*2 + 3]
            lj1 = self.training_samples[current_index, neg_index*2 + 4]
            yield xu1, xi1, li1, xj1, lj1
            
    def get_validation_batch(self, BATCH_SIZE):
        rtn = []
        for i in range(0, self.validation_samples.shape[0], BATCH_SIZE):
            xu1 = self.validation_samples[i:(i+BATCH_SIZE), 0]
            xi1 = self.validation_samples[i:(i+BATCH_SIZE), 1]
            xl1 = self.validation_samples[i:(i+BATCH_SIZE), 2]
            xj1 = self.validation_samples[i:(i+BATCH_SIZE), 3]
            rtn.append([xu1, xi1, xl1, xj1])
        return rtn

    def model_constructor(self, n_user, n_item, n_stage, HIDDEN_DIM, LAMBDA, LEARNING_RATE):
        
        u = tf.placeholder(tf.int32, [None])
        i = tf.placeholder(tf.int32, [None])
        j = tf.placeholder(tf.int32, [None])
        
        li = tf.placeholder(tf.int32, [None])
        lj = tf.placeholder(tf.int32, [None])
        li_onehot = tf.one_hot(li, n_stage)
        lj_onehot = tf.one_hot(lj+1, n_stage)
    
        user_emb = tf.get_variable("user_emb", [n_user, HIDDEN_DIM], 
                                   initializer=tf.random_uniform_initializer(-0.01, 0.01))
        item_emb = tf.get_variable("item_emb", [n_item, HIDDEN_DIM], 
                                   initializer=tf.random_uniform_initializer(-0.01, 0.01))
        stage_emb = tf.get_variable("stage_emb", [n_stage, HIDDEN_DIM], 
                                    initializer=tf.random_uniform_initializer(-0.01, 0.01))
        item_bias = tf.get_variable("item_bias", [n_item, 1], initializer=tf.constant_initializer(0))
        user_bias = tf.get_variable("user_bias", [n_user, 1], initializer=tf.constant_initializer(0))
        b0 = tf.get_variable("global_bias", [1], initializer=tf.constant_initializer(0))
        
        u_emb = tf.nn.embedding_lookup(user_emb, u)
        i_emb = tf.nn.embedding_lookup(item_emb, i)
        j_emb = tf.nn.embedding_lookup(item_emb, j)
        
        i_b = tf.nn.embedding_lookup(item_bias, i)
        j_b = tf.nn.embedding_lookup(item_bias, j)
        u_b = tf.nn.embedding_lookup(user_bias, u)
        
        alpha0 = tf.get_variable("alpha", [1], initializer=tf.constant_initializer(1))
        alpha = tf.nn.relu(alpha0) + 1.0
    
        u_emb0 = tf.multiply(tf.reshape(u_emb, [tf.shape(u_emb)[0], 1, HIDDEN_DIM]), 
                             tf.reshape(stage_emb, [1, n_stage, HIDDEN_DIM]))
        item_us_emb = tf.transpose(tf.tensordot(u_emb0, item_emb, axes=[[2],[1]]), [0,2,1])
        item_us_emb = tf.nn.softplus(item_us_emb*alpha)/alpha
        s = tf.cumsum(item_us_emb, axis=2, reverse=True) + tf.reshape(item_bias, [1, n_item, 1])
    
        di = tf.tensordot(tf.multiply(u_emb, i_emb), stage_emb, axes=[[1],[1]])
        dpi = tf.nn.softplus(di*alpha)/alpha
        
        dj = tf.tensordot(tf.multiply(u_emb, j_emb), stage_emb, axes=[[1],[1]])
        dpj = tf.nn.softplus(dj*alpha)/alpha
        
        mask_first = tf.equal(lj, -1)
        l_onehot_prev = tf.one_hot(tf.where(mask_first, tf.zeros_like(lj), lj), n_stage)
        mj = tf.nn.softplus(tf.reduce_sum(tf.multiply(dj, l_onehot_prev), 1, keep_dims=True)*alpha)/alpha
        mj = tf.log(tf.exp(mj) - 1 + 1e-10) - mj
        mj = tf.where(mask_first, tf.zeros_like(mj), mj)
        
        si = tf.multiply(tf.cumsum(dpi, axis=1, reverse=True), li_onehot)
        sj = tf.multiply(tf.cumsum(dpj, axis=1, reverse=True), lj_onehot)
        
        x_pos = tf.reduce_sum(si, 1, keep_dims=True) + i_b + u_b + b0
        x_neg = tf.reduce_sum(sj, 1, keep_dims=True) + j_b + u_b + b0

        l2_norm = tf.add_n([tf.reduce_sum(tf.multiply(u_emb, u_emb)), 
                            tf.reduce_sum(tf.multiply(i_emb, i_emb)),
                            tf.reduce_sum(tf.multiply(j_emb, j_emb))])
                           
        mask_last = tf.equal(lj, n_stage-1)
        neg_loss = tf.where(mask_last, tf.zeros_like(x_neg), - x_neg + tf.log_sigmoid(x_neg)) + mj
    
        logloss = - tf.reduce_sum(tf.log_sigmoid(x_pos) + neg_loss)
        valiloss = - tf.reduce_sum(tf.log_sigmoid(x_pos) - x_neg + tf.log_sigmoid(x_neg))
        logloss0 = LAMBDA*l2_norm + logloss
        
        optimizer = tf.contrib.opt.LazyAdamOptimizer(LEARNING_RATE).minimize(logloss0)
        
        return u, i, j, li, lj, s, logloss, optimizer, valiloss
            
    def train_edgeOpt(self):
        n_user = self.n_user
        n_item = self.n_item
        n_stage = self.n_stage
        
        HIDDEN_DIM = self.HIDDEN_DIM
        LAMBDA = self.LAMBDA
        LEARNING_RATE = self.LEARNING_RATE
        BATCH_SIZE = self.BATCH_SIZE
        MODEL_NAME = self.MODEL_NAME
        
        EPOCHS = 5000
        max_noprogress = 10
        
        batch_validation_final = self.get_validation_batch(BATCH_SIZE)
        

        print("start training "+MODEL_NAME+" ...")
        sys.stdout.flush()
        config = tf.ConfigProto()
        with tf.Graph().as_default(), tf.Session(config=config) as session:
            u, i, j, li, lj, s, logloss, optimizer, valiloss = self.model_constructor(n_user, n_item, n_stage, HIDDEN_DIM, LAMBDA, LEARNING_RATE)
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
                
            _loss_train_min = 1e10
            _loss_vali_min = 1e10
            _loss_vali_old = 1e10
            n_noprogress = 0
            
            for epoch in range(1,EPOCHS):
                count = 0
                count_sample = 0
                _loss_train = 0

                print("=== current batch: ", end="")
                for xu1, xi1, li1, xj1, lj1 in self.next_training_batch_edgeOpt(BATCH_SIZE):
                    _loss_train_batch, _ = session.run([logloss, optimizer], feed_dict={u:xu1, i:xi1, li: li1, 
                                                       j:xj1, lj: lj1})
                    _loss_train += _loss_train_batch
                    count += 1.0
                    count_sample += len(xu1)
                    if count % 500 == 0:
                        print(int(count), end=", ")
                        sys.stdout.flush()
                print("complete!")
                _loss_train /= count_sample
                if _loss_train < _loss_train_min:
                    _loss_train_min = _loss_train
                print("epoch: ", epoch, "  train_loss: {:.4f}, min_loss: {:.4f}".format(_loss_train, _loss_train_min), end=",   ")
    
                count_sample = 0
                _loss_vali = 0.0
                for xu1, xi1, xl1, xj1 in batch_validation_final:
                    _loss_vali_batch = session.run(valiloss, feed_dict={u:xu1, i:xi1, li: xl1, j:xj1, lj: xl1-1})
                    _loss_vali += _loss_vali_batch
                    count_sample += len(xu1)
                _loss_vali /= count_sample
                
                if _loss_vali <= _loss_vali_min:
                    _loss_vali_min = _loss_vali
                    n_noprogress = 0
                    saver.save(session, MODEL_DIR + self.MODEL_NAME + ".model.ckpt")
                if _loss_vali > _loss_vali_old:
                    n_noprogress += 1
                _loss_vali_old = _loss_vali

                
                print("vali_loss: {:.4f}, min_loss: {:.4f}".format(_loss_vali, _loss_vali_min), 
                      "  #no progress: ", n_noprogress)
                sys.stdout.flush()
                if n_noprogress > max_noprogress:
                    break
            saver.restore(session, MODEL_DIR + self.MODEL_NAME + ".model.ckpt")
        print("done!")
        sys.stdout.flush()
        
        
    def evaluate_model_slice(self, u, s, stage_id, data_test_slice, user_item_map, topK):
        n_item = self.n_item
        MODEL_NAME = self.MODEL_NAME
        BATCH_SIZE = 20
        
        print("evaluating model: "+MODEL_NAME+" ...")
        metric_vali_slice = []
        metric_test_slice = []
        print("current progress: ", end="")
        for _k0 in range(0, data_test_slice.shape[0], BATCH_SIZE):
            if _k0 % 5000 == 0:
                print(str(_k0)+"/"+str(data_test_slice.shape[0]), end=", ")
                sys.stdout.flush()
            user_list = data_test_slice[_k0:(_k0+BATCH_SIZE),0]
            _s_list = np.array(s.eval(feed_dict={u:user_list}))
            for _ki in range(len(user_list)):
                _k = _k0 + _ki
                _u = user_list[_ki]
                _s = _s_list[_ki,:]
                item_u = user_item_map[_u]
                item_list = item_u[item_u[:,1]>=stage_id,0]
                _mask = np.zeros(n_item)
                _mask[np.array(item_list)] = 1                    
                _s_neg = _s[_mask<1]
                n = len(_s_neg)
                # validation
                if data_test_slice[_k,2] >= stage_id:
                    _i = data_test_slice[_k,1]
                    wrong = np.sum(_s_neg >= _s[_i])
                    _auc = (n - wrong)/n
                    _ndcg = 1.0/np.log2(2 + wrong)
                    _hr = int(wrong < topK)
                    metric_vali_slice.append([_auc, _ndcg, _hr])
                # test
                if data_test_slice[_k,4] >= stage_id:
                    _i = data_test_slice[_k,3]
                    wrong = np.sum(_s_neg >= _s[_i])
                    _auc = (n - wrong)/n
                    _ndcg = 1.0/np.log2(2 + wrong)
                    _hr = int(wrong < topK)
                    metric_test_slice.append([_auc, _ndcg, _hr])
        print("complete!")
        metric_vali_slice = np.array(metric_vali_slice)
        metric_test_slice = np.array(metric_test_slice)
        print("done!")
        return metric_vali_slice, metric_test_slice
    
        
    def evaluation(self, data_test, user_item_map, topK):
        n_user = self.n_user
        n_item = self.n_item
        n_stage = self.n_stage
        
        HIDDEN_DIM = self.HIDDEN_DIM
        LAMBDA = self.LAMBDA
        LEARNING_RATE = self.LEARNING_RATE
        MODEL_NAME = self.MODEL_NAME

        target_stage_id = self.target_stage_id
        with tf.Graph().as_default(), tf.Session() as session:
            u, i, j, li, lj, s, logloss, optimizer, valiloss = self.model_constructor(n_user, n_item, n_stage, HIDDEN_DIM, LAMBDA, LEARNING_RATE)
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(session, MODEL_DIR + self.MODEL_NAME + ".model.ckpt")
            index_list = (data_test['max_stage_vali']>=target_stage_id) + (data_test['max_stage_test']>=target_stage_id)
            data_test_slice = data_test[index_list].values
            metric_vali, metric_test = self.evaluate_model_slice(u, s, target_stage_id, data_test_slice, user_item_map, topK=topK)

        res = np.array([metric_vali.mean(axis=0), metric_test.mean(axis=0)])
        print("AUC={:.3f}, NDCG={:.3f}, HR={:.3f} (validation)".format(res[0,0], res[0,1], res[0,2]))
        print("AUC={:.3f}, NDCG={:.3f}, HR={:.3f} (test)".format(res[1,0], res[1,1], res[1,2]))
        np.savetxt(OUTPUT_DIR + MODEL_NAME + ".stage"+str(target_stage_id) +".result.csv", res, delimiter=", ")
        