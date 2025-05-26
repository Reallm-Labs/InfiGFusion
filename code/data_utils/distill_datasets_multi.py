import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
import blosc
import random

from utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer


class DistillDataset(Dataset):
    def __init__(
        self, 
        args, 
        split: str,
        student_tokenizer: Dict[str, AutoTokenizer],
        teacher_tokenizers: Optional[Dict[str, AutoTokenizer]] = {},
    ):
        self.args = args
        # bind teacher_model_name with model_type
        self.teacher_model_name_and_type = {}
        teacher_model_names = args.teacher_model_name.split(':')
        teacher_model_types = args.teacher_model_type.replace('.', '_').split(':')
        for model_type, model_name in zip(teacher_model_types, teacher_model_names):
            self.teacher_model_name_and_type[model_type] = model_name

        # check the consistency between model_types and teacher_tokenizers 
        teacher_tokenizer_keys = set(teacher_tokenizers.keys())
        args_model_types = set(teacher_model_types)

        if teacher_tokenizer_keys == args_model_types:
            print("Keys are consistent!")
        else:
            missing_in_teacher = args_model_types - teacher_tokenizer_keys
            missing_in_args = teacher_tokenizer_keys - args_model_types
            if missing_in_teacher:
                print(f"These types are in args but not in teacher_tokenizers: {missing_in_teacher}")
            if missing_in_args:
                print(f"These types are in teacher_tokenizers but not in args: {missing_in_args}")
        
        # additional for different templates
        if args.diff_templates:
            # Define the available template mapping; ensure the data contains these suffixes.
            self.available_template_mapping = {
            'phi': 'phi-4',
            'phi4': 'phi-4',
            'qwen': 'qwen',
            'qwen14coder': 'qwen',
            'qwen1': 'qwen',
            'mistral': 'Mistral-Small-24B-Instruct',
            'mistralai': 'Mistral-Small-24B-Instruct'
            }
            print(f"Available Templates:\n{self.available_template_mapping}")

            # Parse teacher template suffixes from args and strip any whitespace.
            teacher_template_suffixes = [item.strip() for item in self.args.tec_templates_suffixes.split(",")]
            print("Teacher Templates:")
            self.teacher_templates_mapping = {}
            for suffix in teacher_template_suffixes:
                if suffix in self.available_template_mapping:
                    template = self.available_template_mapping[suffix]
                    self.teacher_templates_mapping[suffix] = template
                    print(f"  Suffix: {suffix}, Template: {template}")
                else:
                    print(f"  Suffix: {suffix}, Template: Not found")

            # Ensure that all provided teacher template suffixes are valid
            assert len(self.teacher_templates_mapping) == len(teacher_template_suffixes), \
            "Some teacher templates are not found in the available template mapping"
            assert len(self.teacher_templates_mapping) == len(teacher_tokenizers), \
            "The number of teacher templates should equal the number of teacher tokenizers"

            # Retrieve and print the student template using the provided suffix.
            self.student_template_suffix = self.available_template_mapping[self.args.stu_template_suffix]
            print("Student Template:")
            print(f"  Suffix: {self.args.stu_template_suffix}, Template: {self.student_template_suffix}")

            # Map teacher model types to their corresponding template suffix.
            self.template_suffix_to_model_type = dict(zip(teacher_model_types, teacher_template_suffixes))
            print(f"Template suffix mapping for model types: {self.template_suffix_to_model_type}")

        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.args.data_dir_list = self.args.data_dir.split(",")                                                                  8        
        self.args.feats_folder_list = self.args.feats_folder.split(",") 
        self.dataset = self._load_and_process_data()

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_tokenize_same_temp(self, raw_data, _feats_folder):
        dataset = []
        self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in raw_data]
        
        if self.args.debug:
            log_rank('For debug, only load small parts of data')
            raw_data = raw_data[:100]
            self.answers = self.answers[:100]
        # scaling sampling dataset, usually full data
        if self.args.data_scale_factor < 1.0:
            ori_data_num = len(raw_data)
            new_data_num = int(ori_data_num * self.args.data_scale_factor)
            sample_indices = random.sample(range(ori_data_num), new_data_num)
            # 使用相同的索引同时更新 raw_data 和 self.answers
            raw_data = [raw_data[i] for i in sample_indices]
            self.answers = [self.answers[i] for i in sample_indices]
            
        log_rank("Processing dataset for student model (and all teacher models)...")  
        seg = np.iinfo(np.int32).max * 2 + 1      # [seg] has been removed in `process_lm`
        cur_idx = 0
        for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
            student_prompt_ids = self.student_tokenizer.encode(
                data["prompt"], add_special_tokens=False
            )
            student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
            student_response_ids = self.student_tokenizer.encode(
                data["output"], add_special_tokens=False
            )
            # add eos behind each response
            student_response_ids = student_response_ids \
                                    + [self.student_tokenizer.eos_token_id]
            tokenized_data = {
                "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
            }
    
            # support teachers with different tokenziers
            for model_type, tokenizer in self.teacher_tokenizers.items():
                if tokenizer is None:
                    continue # for tokenizers that are not initilized
                    
                teacher_prompt_ids = tokenizer.encode(
                    data["prompt"], add_special_tokens=False
                )
                teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                teacher_response_ids = tokenizer.encode(
                    data["output"], add_special_tokens=False
                )
                teacher_response_ids = teacher_response_ids \
                                        + [tokenizer.eos_token_id]
                tokenized_data[f"teacher_{model_type}_input_ids"] = \
                    teacher_prompt_ids + [seg] + teacher_response_ids
                tokenized_data["feats_folder"]= _feats_folder

            # additional add `index` for later loading teacher hidden states and logits
            if self.args.data_scale_factor < 1.0:
                tokenized_data[f"data_index"] = sample_indices[cur_idx]
            else:
                tokenized_data[f"data_index"] = cur_idx
            cur_idx += 1
            dataset.append(tokenized_data)
        return dataset
    def _load_and_tokenize_diff_temp(self, raw_data, _feats_folder):
        dataset = []
 
        # for multi templates, the student gt is geneerated_solution (which is the typo, not fix now)
        self.answers = [x["generated_solution"] if isinstance(x["generated_solution"], list) else [x["generated_solution"]] for x in raw_data]
        
        if self.args.debug:
            log_rank('For debug, only load small parts of data')
            raw_data = raw_data[:100]
            self.answers = self.answers[:100]
        # scaling sampling dataset, usually full data
        if self.args.data_scale_factor < 1.0:
            ori_data_num = len(raw_data)
            new_data_num = int(ori_data_num * self.args.data_scale_factor)
            sample_indices = random.sample(range(ori_data_num), new_data_num)
            # 使用相同的索引同时更新 raw_data 和 self.answers
            raw_data = [raw_data[i] for i in sample_indices]
            self.answers = [self.answers[i] for i in sample_indices]
            
        log_rank("Processing dataset for student model (and all teacher models)...")  
        seg = np.iinfo(np.int32).max * 2 + 1      # [seg] has been removed in `process_lm`
        cur_idx = 0
        for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):

            student_prompt_ids = self.student_tokenizer.encode(
                data[f"prompt_{self.student_template_suffix}"], add_special_tokens=False
            )
            student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
            # for multi templates, the student gt is geneerated_solution (which is the typo, not fix now)
            student_response_ids = self.student_tokenizer.encode(
                data["generated_solution"], add_special_tokens=False
            )
            # add eos behind each response
            student_response_ids = student_response_ids \
                                    + [self.student_tokenizer.eos_token_id]
            tokenized_data = {
                "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
            }
    
            # support teachers with different tokenziers
            for model_type, tokenizer in self.teacher_tokenizers.items():
                tec_template_str = self.template_suffix_to_model_type[model_type]
                cur_tec_template = self.teacher_templates_mapping[tec_template_str]
                if tokenizer is None:
                    continue # for tokenizers that are not initilized
                    
                teacher_prompt_ids = tokenizer.encode(
                    data[f"prompt_{cur_tec_template}"], add_special_tokens=False
                )
                teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                teacher_response_ids = tokenizer.encode(
                    data[f"output_{cur_tec_template}"], add_special_tokens=False
                )
                teacher_response_ids = teacher_response_ids \
                                        + [tokenizer.eos_token_id]
                tokenized_data[f"teacher_{model_type}_input_ids"] = \
                    teacher_prompt_ids + [seg] + teacher_response_ids
                tokenized_data["feats_folder"]= _feats_folder

            # additional add `index` for later loading teacher hidden states and logits
            if self.args.data_scale_factor < 1.0:
                tokenized_data[f"data_index"] = sample_indices[cur_idx]
            else:
                tokenized_data[f"data_index"] = cur_idx
            cur_idx += 1
            dataset.append(tokenized_data)

    
        return dataset
    
    def _load_and_process_data(self):
        dataset = []
        for _data_dir, _feats_folder in zip(self.args.data_dir_list, self.args.feats_folder_list):

            path = os.path.join(_data_dir, f"{self.split}.jsonl")
            assert os.path.exists(path), f"Error: The path '{path}' does not exist!"
            with open(path) as f:
                raw_data = [json.loads(l) for l in f.readlines()]
            
            if self.args.diff_templates:
                dataset.extend(self._load_and_tokenize_diff_temp(raw_data, _feats_folder))
            else:
                dataset.extend(self._load_and_tokenize_same_temp(raw_data, _feats_folder))   

        
        return dataset


        
    def _process_lm(
        self, i, samp, model_data, no_model_data, gen_data, 
        teacher_model_data, teacher_no_model_data
    ):
        seg = np.iinfo(np.int32).max * 2 + 1
        input_ids = np.array(samp["student_input_ids"])
        source_len = np.where(input_ids == seg)[0][0]
        prompt = input_ids[:source_len]
        # remove seg, concat prompt and response into one sequence
        input_ids = np.concatenate(
            [input_ids[:source_len], input_ids[source_len+1:]], axis=0
        )
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        # yes, the input_ids can reach the max length to the second token to last, as the last token need to be predicted
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        # additional add data index for offline teacher features/logits loading
        model_data["data_index"][i] = samp["data_index"]
        
        # no model data means "not inputted to the model", so usually are label, loss, etc
        # yes, label is from input_ids[1] -> the last token of input_ids
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        # to ignore the loss on the prompt
        no_model_data["label"][i][:source_len-1] = -100
        # also set loss_mask to ensure this, ignoring the loss on the prompt
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        no_model_data["loss_mask"][i][:source_len-1] = 0
        
        # it is for inference, when the data is left-padded, the the newly generated token will
        # be right after the prompt, which is more efficient for autogressive task inference
        gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
        gen_data["attention_mask"][i][-len(prompt):] = 1.0

        cur_file_path = os.path.dirname(os.path.abspath(__file__))
        for model_type in self.teacher_tokenizers:
            # for multiple teacher data loading
            cur_data_index = samp["data_index"]
            folder_index = int(cur_data_index) // 1000
            # data_path = os.path.join("/s1/wangyuanyi/", self.args.feats_folder, \
            #    f"{self.teacher_model_name_and_type[model_type]}_{self.args.merge_folder}", 
            #    f"{folder_index:05d}", f"{cur_data_index:08d}.npz")
            data_path = os.path.join(samp['feats_folder'], \
               f"{self.teacher_model_name_and_type[model_type]}_{self.args.merge_folder}", 
               f"{folder_index:05d}", f"{cur_data_index:08d}.npz")

            
            teacher_feats_data = np.load(data_path, allow_pickle=True)
            # decompress the data and convert to pytorch tensor
            indices = teacher_feats_data["indices"]
            hiddens = teacher_feats_data["hiddens"]
            hiddens_shape = teacher_feats_data["hiddens_shape"]
            input_emb_state = teacher_feats_data.get("input_emb_state", None)
            input_emb_shape = teacher_feats_data.get("input_emb_shape", None)
            target_emb_state = teacher_feats_data.get("target_emb_state", None)
            target_emb_shape = teacher_feats_data.get("target_emb_shape", None)
            def decompress_and_convert(compress_data, data_shape, dtype=torch.bfloat16):
                if compress_data is None:
                    return None
                decompressed_data = np.frombuffer(blosc.decompress(compress_data), dtype=np.float16).reshape(data_shape)
                return torch.from_numpy(decompressed_data.reshape(data_shape)).to(dtype)
            teacher_feats = {
                "indices": indices,  # Indices do not require decompression
                "hiddens": decompress_and_convert(hiddens, hiddens_shape),
                "input_emb_state": decompress_and_convert(input_emb_state, input_emb_shape),
                "target_emb_state": decompress_and_convert(target_emb_state, target_emb_shape)
            }

            # add additional teacher data into teacher_model_data[model_type]
            for teacher_feats_key, teacher_feats_value in teacher_feats.items():
                index_key = f"{cur_data_index:08d}_"
                teacher_model_data[model_type][index_key+teacher_feats_key] = teacher_feats_value

            t_input_ids = np.array(samp[f"teacher_{model_type}_input_ids"])
            t_source_len = np.where(t_input_ids == seg)[0][0]
            t_input_ids = np.concatenate(
                [t_input_ids[:t_source_len], t_input_ids[t_source_len+1:]], axis=0
            )
            t_input_ids = t_input_ids[:self.max_length]
            t_input_len = len(t_input_ids)
            teacher_model_data[model_type]["input_ids"][i][:t_input_len-1] = \
                torch.tensor(t_input_ids[:-1], dtype=torch.long)
            teacher_model_data[model_type]["attention_mask"][i][:t_input_len-1] = 1.0
            if model_type in ["gpt2"]:
                teacher_model_data[model_type]["position_ids"][i][:t_input_len-1] = \
                    torch.arange(0, t_input_len-1, dtype=torch.long)
            teacher_no_model_data[model_type]["label"][i][:t_input_len-1] = \
                torch.tensor(t_input_ids[1:], dtype=torch.long)
            teacher_no_model_data[model_type]["label"][i][:t_source_len-1] = -100
            teacher_no_model_data[model_type]["loss_mask"][i][:t_input_len-1] = 1.0
            teacher_no_model_data[model_type]["loss_mask"][i][:t_source_len-1] = 0

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)
                elif isinstance(data[k], dict):
                    for kk in data[k]:
                        data[k][kk] = data[k][kk].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        # it contains `bz` dimension.
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                        * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, max_length),
            # addition index, useful for offline teacher hidden features and logits usage
            "data_index": torch.ones(bs, 1, dtype=torch.long)
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) \
                        * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        teacher_model_data = {
            model_type: {
                "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                            * self.teacher_tokenizers[model_type].eos_token_id,
                "attention_mask": torch.zeros(bs, max_length),
            } for model_type in self.teacher_tokenizers
        }

        for model_type in self.teacher_tokenizers:
            if model_type in ["gpt2"]:
                teacher_model_data[model_type]["position_ids"] = torch.zeros(
                    bs, max_length, dtype=torch.long
                )

        teacher_no_model_data = {
            model_type: {
                "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
                "loss_mask": torch.zeros(bs, max_length),
            } for model_type in self.teacher_tokenizers
        }

        # ok, firstly provide placeholder data, and then put true data into placeholders via process_lm
        for i, samp in enumerate(samples):
            self._process_lm(
                i, samp, model_data, no_model_data, gen_data, 
                teacher_model_data, teacher_no_model_data
            )

        # add prefix `teacher_{model_type}_` for data from teachers
        for model_type in teacher_model_data:
            prefix = f"teacher_{model_type}_"
            for key in teacher_model_data[model_type]:
                model_data[f"{prefix}{key}"] = teacher_model_data[model_type][key]
                
            for key in teacher_no_model_data[model_type]:
                no_model_data[f"{prefix}{key}"] = teacher_no_model_data[model_type][key]
        
        return model_data, no_model_data, gen_data
   
