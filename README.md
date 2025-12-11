# Deep Active Learning with Formal Verification

## Description

This repository contains the implementation of the paper On Improving Deep Active Learning with Formal Verification by
Spiegelman, Amir and Katz. It contains the code used for the experiments in the paper.
Our paper introduced a novel data augmentation technique for Deep Active Learning (DAL) that
utilizes formal verification to generate diverse adversarial inputs. Unlike standard gradient-based attacks
(e.g., FGSM), this method systematically explores the input neighborhood to uncover multiple distinct counterexamples,
significantly enriching the training set and improving model generalization.

## Technical Details

The project assumes an installation of the Marabou verifier (https://github.com/NeuralNetworkVerification/Marabou),
with a path set up inside the `get_advs.py` file.

The code was run on a cluster with Slurm. The main file for running a Slurm job is `SendToCluster.sbatch`. Usage
example:

sendToCluster.sbatch MNIST_dec_11 MNIST --methods "rand badge FGSM_w_samples_10" 0 20 50

This will run an experiment on MNIST with the Random, BADGE and FVAAL methods, with 20 cycles and 50 queried samples
each cycle. The `SendToCluster.sbatch` file can be edited to specifically set other arguments. 

Valid DAL methods:
- "rand" - random sampling
- "badge" - BADGE sampling
- "DFAL" - DeepFool Active Learning
- "FGSM" - choosing based on binary search on FGSM attacks

To add data augmentation, one of two suffixes should be added:
- "_w_samples_<k>" - adding <k> samples based on the Marabou verifier ("+FV-Adv" in the paper)
- "_FGSM_bounded_<k>" - adding <k> samples based on FGSM attacks in the bounded range [0.05,0.1] ("+FGSM-Adv" in the paper)

For example, the FVAAL method mentioned in the paper can be run with the name "FGSM_w_samples_10"