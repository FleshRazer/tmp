import os


llama2_checkpoint_s_filename = "tmp/models/llama2-7b/checkpoint-17000-sharded.zip"
llama2_checkpoint_filename = "tmp/models/llama2-7b/checkpoint-17000.zip"
llama2_final_s_filename = "tmp/models/llama2-7b/final-sharded.zip"
llama2_final_filename = "tmp/models/llama2-7b/final.zip"

filtered_paranmt_filename = "tmp/data/raw/filtered_paranmt.zip"


def setup_repository():
    os.system(f"zip -F {llama2_checkpoint_s_filename} --out {llama2_checkpoint_filename}")
    os.system(f"unzip {llama2_checkpoint_filename} -d {os.path.dirname(llama2_checkpoint_filename)}")

    os.system(f"zip -F {llama2_final_s_filename} --out {llama2_final_filename}")
    os.system(f"unzip {llama2_final_filename} -d {os.path.dirname(llama2_final_filename)}")

    os.system(f"unzip {filtered_paranmt_filename} -d {os.path.dirname(filtered_paranmt_filename)}")
    