import numpy as np
import matplotlib.pyplot as plt


def on_click(event, coords, num_clicks, verbose=False):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    if verbose:
        print("x: "+str(ix)+" y: "+str(iy))

    coords.append([ix, iy])

    if len(coords) == num_clicks:
        fig.canvas.mpl_disconnect(cid)
        plt.close()

    return coords


def find_nearest(coords, xdata=None, ydata=None):
    nearest_points = []
    for i, coord in enumerate(coords):
        diff_x = np.subtract(xdata, coord[0])
        diff_y = np.subtract(ydata, coord[1])
        diff_total = np.add(diff_x, diff_y)
        diff_mag = np.abs(diff_total)
        near_points.append()



