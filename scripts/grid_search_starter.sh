#!/bin/sh
# (1) main working directory
# shellcheck disable=SC2164
main_wd="$(cd "$PWD"; cd ..; pwd)"
# (2) Script in (1)
python_script_path="$main_wd/main.py"

# shellcheck disable=SC2043
for kgname in "UMLS"
do
  kg_path="$main_wd/KGs/$kgname"
  for model in "DistMult" "ComplEx"
  do
    for epoch in 100
    do
      for dim in 32 64 128 256 512
      do
          # shellcheck disable=SC2154
          config_name="$kgname-$model-$epoch-$dim"
          echo "Running $config_name.log"
          #/bin/bash "$PWD/config_runner.sh" "$python_script_path" "$kg_path" "$model" "$epoch" "$dim" > "$config_name.log"
          python -u "$python_script_path" --path_dataset_folder "$kg_path" --model "$model" --num_epochs "$epoch" --embedding_dim "$dim" --scoring_technique "KvsAll" > "$config_name.log"
          echo "Done!"
      done
    done
  done
done
