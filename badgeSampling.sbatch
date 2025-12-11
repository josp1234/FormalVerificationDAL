#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$4_$1_$2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=short
#SBATCH --time=1:0:0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --output=stdouts/create_row_trial_$1/$4_cycle_$2.out
#SBATCH --exclude=wadi-[01-05],gringolet-03,gringolet-04,gringolet-[01-06],creek-03,dumfries-[001-010]

python3 badge_sampling.py $@
EOT
