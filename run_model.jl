using Base: Float64, Int64
################################################################################
# Run 2d fluid model
################################################################################

using Printf, PyPlot, LinearAlgebra, Random
close("all")
include("helper_functions.jl")
include("struct_defs.jl")
Random.seed!(123);


################################################################################
# The input is the relax = 1/τ and q_jet parameters
################################################################################


function Barotropic(relax = 0.02, jet_type = "2_gaussian_jets", 
                                  jet_params = nothing)

    # Coriolis force
    beta = 0.0
    # horizontal mean/backgroud flow
    ubar = 0.0 

    widthx, widthy = 8*pi, 4*pi 
    nkx = 2^6


    t_max, dt_diag, dt_chkpt = 1000.0, 0.5, 25.0 
    
    n_tracers = 1
    fdir = "./data/"
    
    param = Param(relax=relax, beta=beta,
    widthx=widthx, widthy=widthy, ubar=ubar,
    t_max=t_max, dt_diag=dt_diag, dt_chkpt=dt_chkpt,
    nkx=nkx, n_tracers = n_tracers, 
    fdir=fdir)
    
    nky, nx, ny = param.nky, param.nx, param.ny
    dataf = param.dataf
    dt = param.dt
    nvisc = param.nvisc
    
    # initialize wavenumbers
    wavnum = Wavenumbers(param)
    x, y = wavnum.x, wavnum.y
    # initialize forcing
    gvort_jet = forcing_init(param, wavnum, jet_type, jet_params)
    # initialize stirring
    stirring = stirring_init(param, wavnum)
    
    
    
    # if isfile(dataf)
    #     error(string("File ", dataf, " already exists.")) 
    # else
    #     fid = open(dataf, "w")
    #     println(string("\nWriting data file:    ", dataf))
    # end
    fid = open(dataf, "w")
    println(string("\nWriting data file:    ", dataf))
    
    # allocate memory for relative vorticity and tracers, their time 
    # tendencies (at 3 time leves), and their forcings
    wtracers = complex(zeros(n_tracers, nkx, nky))
    wtracer_tends = complex(zeros(n_tracers, 3, nkx, nky))
    gforcings = zeros(n_tracers, nx, ny)
    
    
    # complete forcing struct
    forcing = Forcing(gvort_jet, gforcings)
    
    
    # initialize time level indices
    tlev = [1 2 3]
    t = [0]
    
    model = Model(param, wavnum, forcing, stirring, wtracers, wtracer_tends, tlev, t)
    
    # write important parameters and initial state to snapshot file
    write(fid, relax)
    write(fid, beta) 
    write(fid, widthx) 
    write(fid, widthy) 
    write(fid, ubar) 
    write(fid, dt_diag) 
    write(fid, dt_chkpt) 
    write(fid, dt) 
    write(fid, nvisc) 
    write(fid, nkx) 
    write(fid, nky) 
    write(fid, nx) 
    write(fid, ny)
    write(fid, n_tracers)

    t_last_diag = model.t[1]
    
    
    # start integration with one forward step ...
    take_step(model, "single")
    # ... and one AB2 step
    take_step(model, "AB2")
    
    
    # plot 
    plot_type = "rel. vort." 
    
    if plot_type == "rel. vort."
        grelvort = gfft2(model.wtracers[1, :, :], nx, ny)
        ph, ax = init_plot(x, y, grelvort, "RdBu_r")
    end
    
    # begin integration
    println(@sprintf("\nCurrent parameter settings:"))
    println(@sprintf("\tbeta  = %6.3f", beta))
    println(@sprintf("\trelax = %10.3e", relax))
    println(@sprintf("\tdt    = %10.3e", dt))
    println(@sprintf("\tt_max = %6.1f", t_max))
    println(@sprintf("\nStarting integration..."))
    while model.t[1] < t_max
        # global t_last_diag
        
        # take a time step
        take_step(model, "AB3")
        
        if (model.t[1] - t_last_diag) >= dt_diag - sqrt(eps())
            if any(isnan.(model.wtracers))
                error("Floating point exception.")
            end
            
            println(@sprintf("t = %9.3f", model.t[1]))
            snapshot(fid, model)
            #= println(@sprintf("avg cond rate: %1.3e", mean(cond)/dt)) =#
            t_last_diag = model.t[1]
            
            if plot_type == "rel. vort."
                grelvort = gfft2(model.wtracers[1, :, :], nx, ny)
                ph.set_array(grelvort[1:end-1, 1:end-1][:])
                vmax = maximum(abs.(grelvort))
                ph.set_clim(vmin=-vmax, vmax=vmax)
                ax.set_title(@sprintf("%s at t=%3.2f", plot_type, model.t[1]))
            end
            
            
        end
        
        if plt.get_fignums() != [1] && plot_type != "none"
            # plot closed, exit
            break
        end
        
    end
    close(fid)
    
    ################################################################################
    # end
    ################################################################################
end

Barotropic()