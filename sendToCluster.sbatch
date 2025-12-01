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

python3 main.py $1 $2 --delete $3 --methods $4 --emb_size $5 --start_cycle $6 --cycles $7 --retrain_only $8 --num_samples $9
EOT
