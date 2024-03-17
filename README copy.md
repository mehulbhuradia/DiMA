# Enzymix
#file transfer
scp -r processed_500 daic:~/Enzymix/


<!-- diffab data -->
scp -r all_structures daic:/tudelft.net/staff-umbrella/Enzymix/diffab/data/

scp -r zipz daic:/tudelft.net/staff-umbrella/Enzymix/LatentDiff/

scp -r LatentDiff daic:/tudelft.net/staff-umbrella/Enzymix/

scp -r af_structures daic:/tudelft.net/staff-umbrella/Enzymix/se3_diffusion/

scp -r uniprot_sprot.fasta daic:/tudelft.net/staff-umbrella/Mehul/Enzymix/

scp -r data/uniprot daic:/tudelft.net/staff-umbrella/Mehul/DiMA/data/
scp -r data/uniprot_trim daic:/tudelft.net/staff-umbrella/Mehul/DiMA/data/
scp -r ESP daic:/tudelft.net/staff-umbrella/Mehul/DiMA/

#setup
module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

cd /tudelft.net/staff-umbrella/Mehul/DiMA

<!-- download log files -->
scp -r daic:/tudelft.net/staff-umbrella/Mehul/DiMA/checkpoints/decoder-esm2-150M-uniprot_500--1000.pth ./
scp -r daic:/tudelft.net/staff-umbrella/Mehul/DiMA/data/ ./

<!-- cuz limited space -->
conda config --add pkgs_dirs /tmp/
