
dest=/home/dna/py_workspace/inverse_rl/data/
src=dna@172.30.0.250:/home/dna/python_ws/inverse_rl/data/cartpole_data_rllab_trpo

/usr/bin/rsync -Pa -e 'ssh -i /mnt/recogizer/dali/.ssh/id_rsa' $src $dest

echo '-------------------------------------------'

src=dna@172.30.0.250:/home/dna/python_ws/inverse_rl/data/cartpole_data_rllab_ppo

/usr/bin/rsync -Pa -e 'ssh -i /mnt/recogizer/dali/.ssh/id_rsa' $src $dest

# echo '-------------------------------------------'

# src=dna@172.30.0.250:/home/dna/python_ws/inverse_rl/inverse_rl
# dest=/mnt/recogizer/dali/inverse_rl/

# /usr/bin/rsync -Pa -e 'ssh -i /mnt/recogizer/dali/.passlessSSH/id_rsa' $src $dest


# echo '-------------------------------------------'