#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=advs_FGSM_$1_$2_$3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=short
#SBATCH --time=12:0:0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --output=stdouts/get_advs_trial_$1/stage_$2_$3_FGSM.out
#SBATCH --exclude=wadi-[01-05],gringolet-03,gringolet-04,gringolet-[01-06],creek-03

python3 get_FGSM_advs.py $@
EOT
