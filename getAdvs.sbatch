#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=advs_$1_$2_$3_$4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=short
#SBATCH --time=12:0:0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --output=stdouts/get_advs_trial_$1/stage_$2_$4_job_$3.out
#SBATCH --exclude=wadi-[01-05],gringolet-03,gringolet-04,gringolet-[01-06],creek-03,dumfries-[001-010]


python3 get_advs.py $@
EOT
