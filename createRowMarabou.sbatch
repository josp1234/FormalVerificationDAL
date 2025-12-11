#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=cr_m_$1_$2_$3
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=short
#SBATCH --time=12:0:0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --output=stdouts/create_row_trial_$1/marabou_cycle_$2_idx_$3.out
#SBATCH --exclude=wadi-[01-05],gringolet-03,gringolet-04,gringolet-[01-06],creek-03,dumfries-[001-009]

python3 create_row_marabou.py $@
EOT
