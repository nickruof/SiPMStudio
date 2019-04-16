import numpy as np
import matplotlib.pyplot as plt
import keyboard


def connect(fig, event_type, event_function):
    cid = fig.canvas.mpl_connect(event_type, event_function)
    return cid


def disconnect(fig, cid):
    if keyboard.is_pressed("q"):
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)


def on_click(event, cid, coords, verbose=False):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    if verbose:
        print("x: "+str(ix)+" y: "+str(iy))
    coords.append([ix, iy])
    return coords


def find_nearest(coord, xdata=None, ydata=None):
    diff_x = np.subtract(xdata, coord[0])
    diff_y = np.subtract(ydata, coord[1])
    diff_total = np.add(diff_x, diff_y)
    diff_mag = np.abs(diff_total)
    index = np.where(diff_mag == min(diff_mag))[0]
    return [xdata[index], ydata[index]]


def nearest_on_click(event, coords, data, verbose=False):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    coords.append(find_nearest([ix, iy], data[0], data[1]))
    return coords






