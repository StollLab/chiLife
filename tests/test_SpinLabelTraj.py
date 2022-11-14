import numpy as np
import matplotlib.pyplot as plt
import chilife

#
# def test_generation():
#     mv0 = ProEPR.fetch('2mv0')
#     SLTraj = ProEPR.SpinLabelTraj('R1A', 45, 'A', mv0)
#
#
# def test_iter():
#     mv0 = ProEPR.fetch('2mv0')
#     SLTraj = ProEPR.SpinLabelTraj('R1A', 45, 'A', mv0)
#     for SL in SLTraj:
#         print(SL)
#
#
# def test_zip():
#     mv0 = ProEPR.fetch('2mv0')
#     SLTraj1 = ProEPR.SpinLabelTraj('R1A', 45, 'A', mv0)
#     SLTraj2 = ProEPR.SpinLabelTraj('R1A', 55, 'A', mv0)
#     for SL1, SL2 in zip(SLTraj1, SLTraj2):
#         print(SL1, SL2)
#
#
# def test_get_dd():
#     mv0 = ProEPR.fetch('2mv0')
#     SLTraj1 = ProEPR.SpinLabelTraj('R1A', 45, 'A', mv0)
#     SLTraj2 = ProEPR.SpinLabelTraj('R1A', 55, 'A', mv0)
#     r = np.linspace(10, 80, 256)
#     P = ProEPR.distance_distribution(SLTraj1, SLTraj2, r)
#
#     plt.plot(r, P)
#     plt.show()
