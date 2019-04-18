def fit_spectrum(params_data, params=None, display=False):
    bins = np.linspace(start=0, stop=max(params_data),
                       num=int(max(params_data)))
    bin_vals, bin_edges = np.histogram(params_data, bins=bins, density=True)
    fit = []
    guess_coords = []
    guess_params = []
    params_sol = []
    sns.set_style("whitegrid")
    if params is None:
        fig, ax = plt.subplots()
        ax.bar(bin_edges[:-1], bin_vals, edgecolor="none")
        ax.set_xlabel("ADC")
        ax.set_ylabel("Norm Counts")
        plt.show(block=False)
        connect_id = connect(fig, "button_press_event",
                            partial(nearest_on_click, coords=guess_coords, data=[bin_edges, bin_vals[:-1]]))
        print("Click on peak heights to be included in the fit, press q to quit and continue")
        pause()
        disconnect(fig, connect_id)
        plt.close()

        for coords in guess_coords:
            guess_params.append(coords[0])
            guess_params.append(coords[1])
            guess_params.append(17) # hard coded sigma guess
        scale_guess = (guess_params[0]-guess_params[-3])/len(guess_params)/3
        params_sol, _converge = curve_fit(multi_gauss_moyal, A=guess_coords[0][1],
                                     loc=guess_coords[0][0], scale=scale_guess, *guess_params)
        fit = multi_gauss_moyal(bins[:-1], A=params_sol[0], loc=params_sol[1], scale=params_sol[2], *params_sol[3])
    else:
        fit = multi_gauss_moyal(bins[:-1], A=params[0], loc=params[1], scale=params[2], *params[3])
    if display:
        plt.figure()
        plt.bar(bin_vals[:-1], bin_edges, edgecolor="none")
        plt.plot(bins[:-1], fit, color="red")
        plt.xlabel("ADC")
        plt.ylabel("Norm Counts")
        plt.legend(["Data", "Fit"])
    if params is None:
        return params_sol