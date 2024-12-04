set -x
# RTL test
verilator --cc adder.v --exe test_adder.cpp
make -j -C obj_dir -f Vadder.mk Vadder
./obj_dir/Vadder

# synthize
yosys -s adder.ys
cat adder_synth.v
# Placement & Routing
# 在 mac 上面要安裝 openroad 似乎得經過 docker，否則應該建不起來
# 否則就得在 linux 上安裝
# openroad -exit adder_floorplan.tcl
# 參考 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/master/docs/user/BuildWithDocker.md
# Running GUI’s with Docker on Mac OS X -- https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc
# 可以用 yoda lee 的 docker https://github.com/yodalee/ubuntu-icestorm-toolchain/tree/master/docker
 