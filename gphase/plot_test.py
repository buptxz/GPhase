import ternary

## Boundary and Gridlines
scale = 30
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 10)
figure.set_dpi(600)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)
tax.gridlines(color="black", multiple=6)
tax.gridlines(color="blue", multiple=2, linewidth=0.5)

# Set Axis labels and Title
fontsize = 20
tax.set_title("Simplex Boundary and Gridlines", fontsize=fontsize)
tax.left_axis_label("Left label $\\alpha^2$", fontsize=fontsize)
tax.right_axis_label("Right label $\\beta^2$", fontsize=fontsize)
tax.bottom_axis_label("Bottom label $\\Gamma - \\Omega$", fontsize=fontsize)

# Set ticks
tax.ticks(axis='lbr', linewidth=1)

# Remove default Matplotlib Axes
tax.clear_matplotlib_ticks()

ternary.plt.show()