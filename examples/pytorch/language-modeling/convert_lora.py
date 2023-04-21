import torch

from transformers import GPT2_LoRAConfig, GPT2_LoRALMHeadModel, pipeline


config = GPT2_LoRAConfig.from_pretrained("gpt2")
model = pipeline(model="gpt2").model
state_dict = model.state_dict()
new_state_dict = state_dict.copy()


rank = 10
config.lora_rank = rank
for name in [
    "transformer.h.N.attn.c_attn",
    "transformer.h.N.attn.c_proj",
    "transformer.h.N.mlp.c_fc",
    "transformer.h.N.mlp.c_proj",
]:
    print(f"Name {name}")
    weight_name = f"{name}.weight"
    bias_name = f"{name}.bias"
    avg_w = torch.stack([state_dict[weight_name.replace("N", str(i))] for i in range(config.n_layer)], dim=0).mean(
        dim=0
    )
    avg_b = torch.stack([state_dict[bias_name.replace("N", str(i))] for i in range(config.n_layer)], dim=0).mean(dim=0)

    for i in range(config.n_layer):
        lname = name.replace("N", str(i))
        w_name = weight_name.replace("N", str(i))
        b_name = bias_name.replace("N", str(i))

        target_w = state_dict[w_name]
        target_b = state_dict[b_name]

        diff_w = target_w - avg_w
        up_b = target_b - avg_b
        U, S, V = torch.linalg.svd(diff_w, full_matrices=False)

        n = S.shape[0]
        down_w = U[:, :n].matmul(torch.diag(S)[:, :rank])
        up_w = V[:rank, :]
        print((torch.matmul(down_w, up_w) - diff_w).norm())

        new_state_dict[w_name] = avg_w
        new_state_dict[b_name] = avg_b
        new_state_dict[f"{lname}.lora_down_w"] = down_w
        new_state_dict[f"{lname}.lora_down_b"] = torch.zeros((rank,))
        new_state_dict[f"{lname}.lora_up_w"] = up_w
        new_state_dict[f"{lname}.lora_up_b"] = up_b
        new_state_dict[f"{lname}.diag"] = torch.ones((target_w.shape[-1],))


model2 = GPT2_LoRALMHeadModel(config)
missing, unexpected = model2.load_state_dict(new_state_dict)

model2.save_pretrained(f"./starting_rank_{rank}_diag")
