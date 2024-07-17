# Note: you will need the cgo_grid.py plugin (https://pymolwiki.org/index.php/Cgo_grid) to run this pml script
remove not (1bci resn R1M)
hide everything, name NEN
cgo_grid [0, 0, 0], [1, 0, 0], [0, 1, 0], 60, 60, 60, color=raspberry
set stick_transparency, 0.5
color gray, 1bci and name C*
color slate, resn R1M and name C* 

set_view (\
    -0.081599995,    0.055718988,   -0.995105147,\
    -0.996581793,    0.008282896,    0.082185254,\
     0.012821333,    0.998409331,    0.054853156,\
     0.000002872,    0.000041947, -180.932250977,\
     2.034287214,   -1.315982938,   -7.871799946,\
  -575.386230469,  937.248596191,  -20.000000000 )