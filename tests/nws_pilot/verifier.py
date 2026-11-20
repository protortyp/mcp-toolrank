"""Frozen Qwen reranker scorer for the NWS capability experiment."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
          '<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = ("Given a request and one API operation, judge whether that operation can fulfill the request. "
               "Check the action and requested constraints. Mere topical overlap is insufficient.")


class Verifier:
    def __init__(self, model_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, local_files_only=True).to(device).eval()
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        tokens = [self.tokenizer.encode(word, add_special_tokens=False) for word in ("no", "yes")]
        assert all(len(label) == 1 for label in tokens)
        self.head = self.model.lm_head.weight[[label[0] for label in tokens]]

    @torch.inference_mode()
    def score(self, pairs):
        entries = [f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {text}"
                   for query, text in pairs]
        core = self.tokenizer(entries, add_special_tokens=False, truncation=True,
                              max_length=512 - len(self.prefix) - len(self.suffix))["input_ids"]
        inputs = self.tokenizer.pad({"input_ids": [self.prefix + item + self.suffix for item in core]},
                                    padding=True, return_tensors="pt").to(self.device)
        hidden = self.model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
        logits = (hidden @ self.head.T).float()
        return (logits[:, 1] - logits[:, 0]).cpu().numpy()
