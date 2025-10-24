# Denoising Diffusion Probabilistic Model 

Generating Images with a Denoising Diffusion Probabilistic Model (with input perturbation) This project trains a diffusion model to generate 64x64 images and perturbs inputs during training to reduce error accumulation across denoising steps. The method follows a standard DDPM setup but adds noise to the intermediate states to mimic inference-time mistakes and improve sample quality. Results focus on LSUN churches; outputs include both full sampling chains and final images, with noted limits from low resolution and compute.

- Trains a diffusion model on 64x64 LSUN churches; adds input noise to improve sampling.
- Generating_Images_with_a_Denoising_Diffusion_Probabilistic_Model_with_input_perturbation.pdf: short report with method/results.
- main.py: train/sample script (training loop, denoising steps, sample generation).
- setup.py: minimal packaging info.

Usage: python main.py --train | python main.py --sample --ckpt path/to/checkpoint.pt