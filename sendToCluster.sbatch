#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=trial_$1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=medium
#SBATCH --time=4-0:0:0
#SBATCH --output=stdouts/trial_$1_cycles_$6_$7.out
#SBATCH --exclude=wadi-[01-05],gringolet-03,gringolet-04,gringolet-[01-06],creek-03,dumfries-[001-010]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yhonatan.spigelman@mail.huji.ac.il

python3 main.py $1 $2 --methods $3 --start_cycle $4 --cycles $5 --num_samples $6
EOT
