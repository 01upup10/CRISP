import clip
import torch
import torch.nn as nn

class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
    
class CLIP_Prompter(nn.Module):
    def __init__(self, classes, task_step, classes_names, prompt_dim, clip_model='ViT-B/32', device='cuda'):
        super().__init__()
        self.classes = classes # num_classes of all tasks, like [20, 5, 5]
        self.prompt_dim = prompt_dim
        self.device = device
        self.clip_model, preprocess = clip.load(clip_model, device=device)
        self.pool = nn.AvgPool1d(2, stride=2)
        del preprocess
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.learnable_prompts = nn.ModuleList([nn.Embedding(n, prompt_dim) for n in classes])
        self.task_step = task_step
        self.classes_names = classes_names
        with torch.no_grad():
            text_descriptions = classes_names
            text_prompts = ["X "+name+"." for name in text_descriptions]
            self.tokenized_prompts = tokenized_prompts = torch.cat([clip.tokenize(p) for p in text_prompts], dim=0).to(self.device)

            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype) # n 77 512
            self.token_prefix = embedding[:, :1, :]  # SOS n 1 512
            self.token_suffix = embedding[:, 2:, :]  # CLS, EOS n 75 512
        self.text_encoder = TextEncoder(self.clip_model)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
    def get_prompts(self, train=True):
        if train:
            i = self.task_step
            s = sum(self.classes[:i]) if i > 0 else 0
            e = sum(self.classes)
            learnable_prompts = self.learnable_prompts[i].weight.unsqueeze(1).to("cuda") # n 1 512
            if learnable_prompts.dtype != self.clip_model.dtype:
                learnable_prompts = learnable_prompts.to(self.clip_model.dtype)
            prefix = self.token_prefix[s:e]
            suffix = self.token_suffix[s:e]

            prompts = torch.cat([prefix, learnable_prompts, suffix], dim=1)
            tokenized_prompts = self.tokenized_prompts

            prompts = self.text_encoder(prompts, tokenized_prompts[s:e])
            prompts = self.pool(prompts.unsqueeze(1)).squeeze(1)
        else:
            prompts = []
            prefixs = []
            suffixs = []
            tokenized_prompts = self.tokenized_prompts
            for i in range(len(self.classes)):
                s = sum(self.classes[:i]) if i > 0 else 0
                e = sum(self.classes[:i+1])
                cur_prompts = self.learnable_prompts[i].weight.unsqueeze(1).to("cuda")
                cur_prefix = self.token_prefix[s:e]
                cur_suffix = self.token_suffix[s:e]
                cur_prompts = torch.cat([cur_prefix, cur_prompts, cur_suffix], dim=1)
                cur_prompts = self.text_encoder(cur_prompts, tokenized_prompts[s:e])
                cur_prompts = cur_prompts.reshape(-1, 256, 2).mean(dim=-1)
                prompts.append(cur_prompts)
        return prompts
    def frozen_old_prompts(self):
        if self.task_step > 1:
            for i in range(self.task_step-1):
                self.learnable_prompts[i].weight.requires_grad = False
        self.learnable_prompts[-1].weight.requires_grad = True
        print(f"learnable_prompts: {self.learnable_prompts[-1].weight.requires_grad}")

class CLIP_Prompter_WithoutEncoder(nn.Module):
    def __init__(self, classes, task_step, classes_names, prompt_dim, clip_model='ViT-B/32', device='cuda'):
        super().__init__()
        self.classes = classes # list, num_classes of per step
        self.prompt_dim = prompt_dim
        self.device = device
        self.learnable_prompts = nn.ModuleList([nn.Embedding(n, prompt_dim) for n in classes])
        self.task_step = task_step
        self.classes_names = classes_names
    def get_prompts(self):
        i = self.task_step
        prompts = self.learnable_prompts[i].weight.unsqueeze(1).to("cuda")
        prompts = prompts.reshape(-1, 256, 2).mean(dim=-1)
        return prompts
    def frozen_old_prompts(self):
        if self.task_step > 1:
            for i in range(self.task_step-1):
                self.learnable_prompts[i].weight.requires_grad = False
        self.learnable_prompts[-1].weight.requires_grad = True
        print(f"learnable_prompts: {self.learnable_prompts[-1].weight.requires_grad}")
