import os

import torch
import numpy
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import RobertaConfig, RobertaForMaskedLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from utils import convert_examples_to_features, create_examples

class MLM_model():
    def __init__(self, codebert_path, max_source_length, load_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaForMaskedLM, AutoTokenizer
        self.config = config_class.from_pretrained(codebert_path)
        self.tokenizer = tokenizer_class.from_pretrained(codebert_path)
        # length config
        self.max_source_length = max_source_length
        # build model
        self.model = model_class.from_pretrained(codebert_path)
        self.tokenizer.get_vocab()
        if load_model_path is not None:
            print("从...{}...重新加载参数".format(load_model_path))
            self.model.load_state_dict(torch.load(load_model_path))
        self.model.to(self.device)

    def train(self, train_filename, train_batch_size, num_train_epochs, learning_rate, early_stop,
              do_eval, dev_filename, eval_batch_size, output_dir, gradient_accumulation_steps=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        train_examples = create_examples(data_path=train_filename,
                                         max_seq_length=self.max_source_length,
                                         masked_lm_prob=0.15,
                                         max_predictions_per_seq=10,
                                         tokenizer=self.tokenizer)
        train_features = convert_examples_to_features(
            train_examples, self.max_source_length, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_mask_ids = torch.tensor([f.mask_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_mask_ids, all_label_ids)

        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size // gradient_accumulation_steps)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=t_total)

        # Start training
        print("***** 开始训练 *****")
        print("  Num examples = %d", len(train_examples))
        print("  Batch size = %d", train_batch_size)
        print("  Num epoch = %d", num_train_epochs)
        self.model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        count = 0
        for epoch in range(num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, mask_ids, label_ids = batch
                # masked_lm_loss

                # print(input_ids, label_ids)
                loss = self.model(input_ids = input_ids, attention_mask = mask_ids, labels = label_ids).loss
                # print(loss)
                tr_loss += loss.item()
                train_loss = round(tr_loss * gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            if do_eval==True:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = create_examples(data_path=dev_filename,
                                                     max_seq_length=self.max_source_length,
                                                     masked_lm_prob=0.15,
                                                     max_predictions_per_seq=10,
                                                     tokenizer=self.tokenizer)
                    eval_features = convert_examples_to_features(
                        train_examples, self.max_source_length, self.tokenizer)
                    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                    all_mask_ids = torch.tensor([f.mask_ids for f in eval_features], dtype=torch.long)
                    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

                    eval_data = TensorDataset(all_input_ids, all_mask_ids, all_label_ids)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                print("\n***** Running evaluation *****")
                print("  epoch = %d", epoch)
                print("  Num examples = %d", len(eval_examples))
                print("  Batch size = %d", eval_batch_size)

                # Start Evaling model
                self.model.eval()
                eval_loss, batch_num = 0, 0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, mask_ids, label_ids = batch
                    with torch.no_grad():
                        loss = self.model(input_ids = input_ids, attention_mask = mask_ids, labels = label_ids).loss
                    eval_loss += loss.item()
                    batch_num += 1
                # Pring loss of dev dataset
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_ppl': round(numpy.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    print("  %s = %s", key, str(result[key]))
                print("  " + "*" * 20)

                # save last checkpoint
                last_output_dir = os.path.join(output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    count = 0
                    print("  Best ppl:%s", round(numpy.exp(eval_loss), 5))
                    print("  " + "*" * 20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir_ppl = os.path.join(output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir_ppl):
                        os.makedirs(output_dir_ppl)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir_ppl, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                else:
                    count += 1
                    print('early stop step: ', count)
                    if (count == early_stop):
                        print('early stop......')
                        break
