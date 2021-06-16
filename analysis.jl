################################################################################
# Analysis of 2d fluid model output
################################################################################
using Printf, PyPlot
# plt.style.use("~/presentation_plots.mplstyle")
close("all")
# load useful functions
include("helper_functions.jl")
include("struct_defs.jl")


function Analysis()
    ################################################################################
    # load data
    ################################################################################
    # snapshot file
    fdir = "./data/"
    fnam = string("beta_", 0.0, "_relax_", 0.02)
    fnbase = string(fdir, fnam)
    dataf = string(fnbase, ".dat")
    fid = open(dataf, "r")
    
    # parameters from the run
    relax, beta, widthx, widthy, ubar, dt_diag, dt_chkpt, dt, nvisc, nkx, nky, nx, ny, n_tracers = load_pars(fid)
    
    # todo 
    param = Param(relax=relax, beta=beta,
    widthx=widthx, widthy=widthy, ubar=ubar,
    dt_diag=dt_diag, dt_chkpt=dt_chkpt,
    nkx=nkx, n_tracers = n_tracers, 
    fdir=fdir)
    # for filting
    wn = Wavenumbers(param)
    # domain
    x, y = wn.x, wn.y
    
    # plot
    # plot_type = "rel. vort."
    # plot_type = "cond"
    # plot_type = "q" 
    # plot_type = "zonal mean q"
    plot_type = "zonal mean rel. vort."
    
    if plot_type == "rel. vort."
        grelvort = zeros(nx, ny)
        ph, ax = init_plot(x, y, grelvort, "RdBu_r")
        img_name = "relvor"
    elseif plot_type == "zonal mean rel. vort."  
        fig, ax = subplots(1)
        grelvort = zeros(nx, ny)
        ph, = ax.plot(y', zonal_mean(grelvort)', "b-", label=L"$relvor_zm$")
        
        ax.axhline(0, ls="-", lw=1.0, c="k")
        ax.set_ylim([0, 1])
        ax.set_xlabel("y")
        ax.set_ylabel("zonal mean relvor")
        ax.legend(loc="upper right")
        img_name = "relvor_zm"
    end
    
    # show snapshot data
    i = 0
    i_img = 0
    while eof(fid) == false
        # read next data
        i += 1
        t, wtracers = read_snapshot(fid, nkx, nky)
        println(t)
        
        if i % 10 == 0
            # update figure
            if plot_type == "rel. vort."
                grelvort = gfft2(wtracers[1, :, :], nx, ny)
                ph.set_array(grelvort[1:end-1,1:end-1][:])
                vmax = maximum(abs.(grelvort))
                ph.set_clim(vmin=-vmax, vmax=vmax)
            elseif plot_type == "zonal mean rel. vort."
                grelvort = gfft2(wtracers[1, :, :], nx, ny)
                ph.set_data(y', zonal_mean(grelvort)')
            end
            ax.set_title(@sprintf("%s at t=%3.2f", plot_type, t[1]))
            
            # save image
            #= savefig(@sprintf("%s%03d.png", img_name, i_img), dpi=100) =#
            pause(0.001)
            i_img += 1
            
            if plt.get_fignums() != [1]
                # plot closed, exit
                break
            end
        end
    end
    close(fid)
end

Analysis()