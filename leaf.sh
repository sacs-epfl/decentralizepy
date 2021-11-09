echo "[Cloning leaf repository]"
git clone https://github.com/TalwalkarLab/leaf.git

echo "[Installing unzip]"
sudo apt-get install unzip

cd leaf/data/shakespeare
echo "[Generating non-iid data]"
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8 --smplseed 10 --spltseed 10

echo "[Data Generated]"