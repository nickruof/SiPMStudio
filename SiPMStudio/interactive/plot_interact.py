import numpy as np
import matplotlib.pyplot as plt
import keyboard


def connect(fig, event_type, event_function):
    cid = fig.canvas.mpl_connect(event_type, event_function)
    return cid


def disconnect(fig, cid):
    if keyboard.is_pressed("q"):
        fig.canvas.mpl_disconnect(cid)


def on_click(event, cid, coords, verbose=True):
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


def nearest_on_click(event, coords, data, verbose=True):
    global ix, iy
    if keyboard.is_pressed("shift"):
        ix, iy = event.xdata, event.ydata
        coords.append(find_nearest([ix, iy], data[0], data[1]))
    return coords


def zoom(event, axis, base_scale=2.0):
    cur_xlim = axis.get_xlim()
    cur_ylim = axis.get_ylim()
    cur_xrange = (cur_xlim[1] - cur_xlim[0])*0.5
    cur_yrange = (cur_ylim[1] - cur_ylim[0])*0.5
    xdata = event.xdata
    ydata = event.ydata
    if event.button == "up":
        scale_factor = 1/base_scale
    elif event.button == "down":
        scale_factor = base_scale
    else:
        scale_factor = 1
        print(event.button)
    axis.set_xlim([xdata - cur_xrange*scale_factor, xdata + cur_xrange*scale_factor])
    axis.set_ylim([ydata - cur_yrange*scale_factor, ydata + cur_yrange*scale_factor])
    plt.draw()


def pause():
    input("Press a key to continue!")






