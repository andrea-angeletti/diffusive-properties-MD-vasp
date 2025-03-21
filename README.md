calculate_msd.py allows to remove the drifting of the center of mass and compute the total means square displacement MSD from XDATCAR during molecular dynamics simulation in VASP, (in this case for all 8 hydrogen atoms in 128 magnesium) 
asd.py allows after to plot the MSD and compute the correspoding diffusion coefficient :
   python asd.py 1 1000 MSD_FFT.npy 
Please reference our work: https://arxiv.org/abs/2407.21088
