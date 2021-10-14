import tqdm

def tqdm_range(start, stop, step=1, verbose=True, text=None, bar_length=40, position=0):
    hide_bar = True
    if verbose:
        hide_bar = False
    bar_format = f"{{l_bar}}{{bar:{bar_length}}}{{r_bar}}{{bar:{-bar_length}b}}"

    return tqdm.trange(start, stop, step, position=position, disable=hide_bar, desc=text, bar_format=bar_format)
