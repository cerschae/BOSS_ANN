#!/bin/bash
#SBATCH -p gpu -o shell.stdout -e shell.stderr --mail-user=christophernstrerne.schaefer@epfl.ch --mail-type=ALL
# run the simulation

module list

unset PYTHONPATH  
export PATH="/dios/shared/apps/anaconda/bin:$PATH"

python ANN_Layer3.py
