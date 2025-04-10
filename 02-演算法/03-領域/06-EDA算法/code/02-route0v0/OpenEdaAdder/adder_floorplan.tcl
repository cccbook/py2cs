# adder_floorplan.tcl
initialize_floorplan -die_area "0 0 1000 1000" -core_area "100 100 900 900"
read_verilog adder_synth.v
read_liberty my_lib.lib
read_lef my_tech.lef
place_io
global_placement
detailed_placement
global_routing
detailed_routing
write_gds adder.gds
