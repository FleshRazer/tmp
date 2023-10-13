import os


llama2_checkpoint_s_filename = "tmp/models/llama2-7b/checkpoint-17000-sharded.zip"
llama2_checkpoint_filename = "tmp/models/llama2-7b/checkpoint-17000.zip"

llama2_final_s_filename = "tmp/models/llama2-7b/final-sharded.zip"
llama2_final_filename = "tmp/models/llama2-7b/final.zip"

llama2_chat_checkpoint_s_filename = "tmp/models/llama2-7b-chat/checkpoint-2000-sharded.zip"
llama2_chat_checkpoint_filename = "tmp/models/llama2-7b-chat/checkpoint-2000.zip"

filtered_paranmt_filename = "tmp/data/raw/filtered_paranmt.zip"


def setup_repository():
    os.system(f"pip install -q -r tmp/requirements.txt")

    os.system(f"zip -FF {llama2_checkpoint_s_filename} --out {llama2_checkpoint_filename}")
    os.system(f"unzip -o {llama2_checkpoint_filename} -d {os.path.dirname(llama2_checkpoint_filename)}")

    os.system(f"zip -FF {llama2_final_s_filename} --out {llama2_final_filename}")
    os.system(f"unzip -o {llama2_final_filename} -d {os.path.dirname(llama2_final_filename)}")

    os.system(f"zip -FF {llama2_chat_checkpoint_s_filename} --out {llama2_chat_checkpoint_filename}")
    os.system(f"unzip -o {llama2_chat_checkpoint_filename} -d {os.path.dirname(llama2_chat_checkpoint_filename)}")

    os.system(f"unzip -o {filtered_paranmt_filename} -d {os.path.dirname(filtered_paranmt_filename)}")
    