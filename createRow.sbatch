#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=cr_$1_$2_$3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=short
#SBATCH --time=12:0:0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --output=stdouts/create_row_trial_$1/stage_$2_idx_$3.out
#SBATCH --exclude=wadi-[01-05],gringolet-03,gringolet-04,gringolet-[01-06],creek-03,dumfries-[001-010]

python3 create_row.py $@
EOT
