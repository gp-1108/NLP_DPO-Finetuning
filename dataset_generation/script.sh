# Base paths
project_dir="/home/gp1108/Code/Thesis/dataset_generation"
ext_dir="/ext"
script_path="${project_dir}/create_script.py"

# Where to find the documents
raw_pdfs="${ext_dir}/Pedagogy_docs"

# Where to save the dataset
extracted_texts_output="${project_dir}/data/extracted_texts.jsonl"
dialogues_output="${project_dir}/data/dialogues.jsonl"
dpo_dialogues_output="${project_dir}/data/dpo_dialogues.jsonl"

# Where to find the prompts
rules_list="${project_dir}/prompts/rules.txt"
good_answer_and_question_prompt="${project_dir}/prompts/good_answer_and_question_prompt.txt"
choose_rule_prompt="${project_dir}/prompts/choose_rule_prompt.txt"
dialogue_prompt="${project_dir}/prompts/dialogue_prompt.txt"

# Setting up singularity and dataset download
raw_pdfs_zip="${ext_dir}/Pedagogy_docs.zip"
sif_image="${ext_dir}/env.sif"
gdrive_sif_id="1fA1JoyOHvyi-WkR-Or8rOXLzvJ0ScKtH"
gdrive_raw_pdf_zip_id="1OZblsFAKoOUlUo0toUcC1H6PZ4vfc29n"

# Adding user installed packages (to use gdown)
export PATH="$PATH:$(python3 -m site --user-base)/bin"
gdown -O $raw_pdfs_zip $gdrive_raw_pdf_zip_id
gdown -O $sif_image $gdrive_sif_id

# Extracting the raw pdfs
cd $ext_dir
unzip -o $raw_pdfs_zip

# Setting up the env
cd $project_dir # to get the logger in the correct place
OPENAI_API_KEY="<insert key here>"

# Script variables
max_generations=2

# Creating the dataset
singularity exec --no-home -B $ext_dir -B $project_dir --env OPENAI_API_KEY=$OPENAI_API_KEY $sif_image \
    python3 $script_path \
    --raw_pdfs $raw_pdfs \
    --extracted_texts_json $extracted_texts_output \
    --dialogues_json $dialogues_output \
    --dpo_dialogues_json $dpo_dialogues_output \
    --rules_list $rules_list \
    --good_answer_and_question_prompt $good_answer_and_question_prompt \
    --choose_rule_prompt $choose_rule_prompt \
    --dialogue_prompt $dialogue_prompt \
    --max_generations $max_generations
