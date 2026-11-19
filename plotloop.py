# import matplotlib
# matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig, ax = plt.subplots()
x = np.linspace(-5, 5, 200)
xv, yv = np.meshgrid(x, x)
a = 1
z = np.exp(-xv * xv - yv * yv)


im = ax.imshow(z, extent=[-5, 5, -5, 5], aspect="auto")
dt = 0.01
t = 0
plt.show()

while True:
    t += dt
    z = np.exp(-np.abs(xv * xv + yv * yv - 4 * np.sin(t))) 

    im.set_data(z)

    fig.canvas.draw()

    # fill in the axes rectangle
    fig.canvas.flush_events()

