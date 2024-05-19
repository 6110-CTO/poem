TMP_FILE=$(mktemp -q /home/yarinbar/EDML/exps/tmp/XXXXXX)

chmod 777 $TMP_FILE

# echo $TMP_FILE
output_file="jobs/output_$(date +%s%N).txt"
echo output goes to $output_file

# write content to tmp file
echo '#!/bin/bash
'$1 > $TMP_FILE

# srun -c 8 -A galileo -p galileo --gres=gpu:1 --output=/dev/null $TMP_FILE
# sbatch -c 8 --gres=gpu:1 -w plato1 --output=$output_file $TMP_FILE
sbatch -c 8 --gres=gpu:1 -A galileo -p galileo --output=$output_file $TMP_FILE
# sbatch -c 8 --gres=gpu:1 --output=$output_file $TMP_FILE
# -w newton9 -w socrates -w lambda5 --gres=gpu:1
rm $TMP_FILE
