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


<!-- ENZYMIX FOLDER -->
scp -r ESP daic:/tudelft.net/staff-umbrella/Enzymix/DiMA/
scp -r data daic:/tudelft.net/staff-umbrella/Enzymix/DiMA/
scp -r checkpoints daic:/tudelft.net/staff-umbrella/Enzymix/DiMA/


#setup
module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

cd /tudelft.net/staff-umbrella/Mehul/DiMA

<!-- download log files -->
scp -r daic:/tudelft.net/staff-umbrella/Mehul/DiMA/checkpoints/decoder-esm2-150M-uniprot_500--1000.pth ./
scp -r daic:/tudelft.net/staff-umbrella/Mehul/DiMA/data/ ./

<!-- cuz limited space -->
conda config --add pkgs_dirs /tmp/


saving checkpoints takes space even if i delete them



cross: 9853329 DIMA/dima
og_afdb: 9853508 Enzymix/dima (inf)
og_afdb: 9853504 Enzymix/dima (long)

L_eps og afdb Enzymix/dima (Nan or inf detected)