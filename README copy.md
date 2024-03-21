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


runing jobs:  
9847304   general dima_dif mbhuradi PD       0:00      1 (Priority)
9847291   general dima_dif mbhuradi  R    1:42:49      1 gpu20
9847287   general dima_dif mbhuradi  R    2:03:28      1 gpu02
9847282   general dima_dif mbhuradi  R    2:33:09      1 gpu23
9847303   general dima_dif mbhuradi  R       3:08      1 gpu15
9847302   general dima_dif mbhuradi  R       5:08      1 gpu01
9847300   general dima_dif mbhuradi  R      14:08      1 gpu07
9847299   general dima_dif mbhuradi  R      16:08      1 gpu01
9847290   general dima_dif mbhuradi  R    1:13:08      1 gpu04