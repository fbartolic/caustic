from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

from matplotlib import pyplot as plt

import daft

pgm = daft.PGM([5, 5], origin=[0, 0])
# Nodes
pgm.add_node(daft.Node("F_obs", r"$\mathbf F_n$", 0, 0))
pgm.add_node(daft.Node("tE", r"$t_E$", 0, -1))
pgm.add_plate(daft.Plate([0.5, 2.25, 2, 2.25],
        label=r"events $n$"))

# Edges
pgm.render()

plt.show()
# pgm.figure.savefig("weaklensing.pdf")
# pgm.figure.savefig("weaklensing.png", dpi=150)