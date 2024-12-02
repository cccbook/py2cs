

```
brew install verilator
brew install yosys

brew install cmake python boost eigen

git clone https://github.com/YosysHQ/nextpnr.git

cd nextpnr

cmake . -DARCH=ice40
make -j$(nproc)
sudo make install

```
