#!/bin/bash

pip install -q -r tmp/requirements.txt

llama2_7b_dir="tmp/models/llama2-7b"
llama2_7b_chat_dir="tmp/models/llama2-7b-chat"

# zip -FF "{$llama2_7b_dir}/" --out llama2_checkpoint_filename
# unzip -o llama2_checkpoint_filename -d {os.path.dirname(llama2_checkpoint_filename)}

# zip -FF {llama2_final_s_filename --out {llama2_final_filename}
# unzip -o {llama2_final_filename -d {os.path.dirname(llama2_final_filename)}

# zip -FF "${llama2_7b_chat_dir}/checkpoint-2000-sharded.zip" --out "${llama2_7b_chat_dir}/checkpoint-2000.zip"
# unzip -o "${llama2_7b_chat_dir}/checkpoint-2000.zip" -d "${llama2_7b_chat_dir}/checkpoint-2000"

cat "${llama2_7b_chat_dir}/checkpoint-2000.tar.gz."* | tar xzvf - -C $llama2_7b_chat_dir

data_dir="tmp/data/raw"
unzip -o "${data_dir}/filtered_paranmt.zip" -d "${data_dir}"
