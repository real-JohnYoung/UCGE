from utils import set_seed
from model import MLM_model
set_seed()
# 初始化模型
model = MLM_model(codebert_path = '../model/unixcoder-base',
                         max_source_length = 128, load_model_path = None)

# 模型训练
model.train(train_filename ='train_wos.txt', train_batch_size = 64, num_train_epochs = 50, learning_rate = 4e-5, early_stop=3,
            do_eval = True, dev_filename ='dev_wos.txt', eval_batch_size = 64, output_dir ='DAPT_TAPT/wosmodel')
