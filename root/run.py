from utils import set_seed
from model import UniXcoder_Seq2Seq
set_seed()
# # 初始化模型

# model = UniXcoder_Seq2Seq(codebert_path = 'model/unixcoder-base', beam_size = 10, loss_fun = 'In_trust',
#                          max_source_length = 64, max_target_length =64 , load_model_path=None)

# # # 模型训练
# model.train(train_filename ='data/assembly/train.csv', train_batch_size = 64, num_train_epochs = 50, learning_rate = 4e-5, early_stop=3,
#             do_eval = True, dev_filename ='data/assembly/dev.csv', eval_batch_size = 64, output_dir ='assembly_In_trust/valid_output')

# # # 加载微调过的模型
# # model/unixcoder-base
# model = UniXcoder_Seq2Seq(codebert_path = 'model/unixcoder-base', beam_size =10 , loss_fun = 'In_trust',
#                          max_source_length = 64, max_target_length = 64, load_model_path = 'assembly_In_trust/valid_output/checkpoint-best-bleu/pytorch_model.bin')

# # 模型测试
# model.test(test_filename ='data/assembly/test.csv', test_batch_size = 16, output_dir ='assembly_In_trust/test_output')




# # adaptibe
# model = UniXcoder_Seq2Seq(codebert_path = 'adaptive/DAPT_TAPT/wosmodel/checkpoint-best-ppl', beam_size = 10, loss_fun = 'CE',max_source_length = 64, max_target_length =64 , load_model_path=None)

# # # 模型训练
# model.train(train_filename ='data/assembly/train.csv', train_batch_size = 64, num_train_epochs = 50, learning_rate = 4e-5, early_stop=3,
#             do_eval = True, dev_filename ='data/assembly/dev.csv', eval_batch_size = 64, output_dir='assembly_CE_adaptive_only/valid_output')

# # 加载微调过的模型
# model/unixcoder-base
model = UniXcoder_Seq2Seq(codebert_path = 'adaptive/DAPT_TAPT/wosmodel/checkpoint-best-ppl', beam_size =10 , loss_fun = 'CE',max_source_length = 64, max_target_length = 64, load_model_path = 'assembly_In_trust_adaptive_only/valid_output/checkpoint-best-bleu/pytorch_model.bin')

# # 模型测试
# model.test(test_filename ='data/assembly/test.csv', test_batch_size = 16, output_dir ='assembly_CE_adaptive_only/test_output')

#模型推理
flag=""
while(flag!="exit"):
    flag=input("FLAG：")
    NL=input("NL：")
    comment = model.predict(source = NL)
    print("Code: ",comment[0])
    