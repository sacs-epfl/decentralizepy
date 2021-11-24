#!\bin\bash

cd
mkdir -p Gitlab
cd Gitlab
git clone git@gitlab.epfl.ch:risharma/decentralizepy.git
cd decentralizepy
mkdir -p leaf/data/femnist/data/train
mkdir -p leaf/data/femnist/data/test
mkdir -p leaf/data/femnist/per_user_data/train
~/miniconda3/bin/conda remove --name decpy --all
~/miniconda3/bin/conda create -n decpy python=3.9
~/miniconda3/envs/decpy/bin/pip install --upgrade pip --quiet
~/miniconda3/envs/decpy/bin/pip install --editable .\[dev\]
