from huggingface_hub import snapshot_download

# Define the local directory for downloading
# local_dir = "./locuslab/tofu_ft_phi-1.5"
# local_dir = "./locuslab/phi_grad_ascent_1e-05_forget01"
local_dir = "./locuslab/tofu_ft_llama2-7b"

# Download the model snapshot
# model = "locuslab/tofu_ft_phi-1.5"
# model = "locuslab/phi_grad_ascent_1e-05_forget01"
# snapshot_download(repo_id=model, revision="checkpoint-3", local_dir=local_dir)
model = 'locuslab/tofu_ft_llama2-7b'
snapshot_download(repo_id=model, local_dir=local_dir)
