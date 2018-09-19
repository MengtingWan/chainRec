import os
import sys
sys.path.append('./src/')
from dataset import Dataset
from chainRec import chainRec
from bprMF import bprMF
import argparse



DATA_DIR = "./data/"
MODEL_DIR = "./models/"
OUTPUT_DIR = "./results/"

for DIR in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    if not os.path.exists(DIR):
        os.makedirs(DIR)


def test_dataset(DATA_NAME, n_stage, method, embed_size, lbda):
    myData = Dataset(DATA_NAME, n_stage)
    myData.split_train_test(seed=1234, max_validation_test_samples=100000)

    validation_samples = myData.sampling_validation()
    
    if method == "chainRec_uniform":
        training_samples = myData.sampling_training(method="edgeOpt_uniform")
        
        myModel = chainRec(myData.n_user, myData.n_item, myData.n_stage, myData.DATA_NAME)
        myModel.config_global(MODEL_CLASS="chainRec_uniform", HIDDEN_DIM=embed_size, 
                              LAMBDA=lbda, LEARNING_RATE=0.001, BATCH_SIZE=256,
                              target_stage_id=(n_stage-1))
        myModel.load_samples(training_samples, validation_samples)
        myModel.train_edgeOpt()
        
        myModel.evaluation(myData.data_test, myData.user_item_map, topK=10)
        
    elif method == "chainRec_stage":
        training_samples = myData.sampling_training(method="edgeOpt_stage")
        
        myModel = chainRec(myData.n_user, myData.n_item, myData.n_stage, myData.DATA_NAME)
        myModel.config_global(MODEL_CLASS="edgeOpt_stage", HIDDEN_DIM=embed_size, 
                              LAMBDA=lbda, LEARNING_RATE=0.001, BATCH_SIZE=256,
                              target_stage_id=(n_stage-1))
        myModel.load_samples(training_samples, validation_samples)
        myModel.train_edgeOpt()
        
        myModel.evaluation(myData.data_test, myData.user_item_map, topK=10)       
        
    elif method == "bprMF":
        training_samples = myData.sampling_training(method="sliceOpt")
        
        myModel = bprMF(myData.n_user, myData.n_item, myData.n_stage, myData.DATA_NAME)
        myModel.config_global(MODEL_CLASS="bprMF", HIDDEN_DIM=embed_size, 
                              LAMBDA=lbda, LEARNING_RATE=0.001, BATCH_SIZE=256,
                              target_stage_id=(n_stage-1))
        myModel.load_samples(training_samples, validation_samples)
        myModel.train_sliceOpt()
        
        myModel.evaluation(myData.data_test, myData.user_item_map, topK=10) 
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yoochoose', 
                        help="specify a dataset from [yoochoose]")
    parser.add_argument('--method', default='chainRec_uniform', 
                        help="specify a training method from [chainRec_uniform, chainRec_stage, bprMF]")
    parser.add_argument('--nStage', default=2, 
                        help="specify the total number of stages", type=int)
    parser.add_argument('--embedSize', default=16,
                        help="specify embedding size", type=int)
    parser.add_argument('--l2', default=0.1,
                        help="specify the hyper-parameter to control l2 penalty", type=float)
    args = parser.parse_args()
    test_dataset(args.dataset, args.nStage, args.method, args.embedSize, args.l2)
    
    