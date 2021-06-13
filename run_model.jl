using Base: Float64, Int64
################################################################################
# Run 2d fluid model
################################################################################

using Printf, PyPlot, LinearAlgebra, Random
# plt.style.use("~/presentation_plots.mplstyle")
close("all")

# load useful functions/structs
include("helper_functions.jl")
include("struct_defs.jl")


Random.seed!(123);
################################################################################
# integrate model
################################################################################

param = Param()
# initialize wavenumbers
wavnum = wavnum_init(param)

# initialize forcing
gvort_jet = forcing_init(wavnum, "point_jet")

# initialize stirring
stirring = stirring_init(param, wavnum)

init, dataf = param.init, param.dataf
x, y = param.x, param.y


relax, beta, widthx, widthy, ubar, dt_diag, dt_chkpt, alpha, dt, nvisc, nkx, nky, nx, ny, n_tracers = 
param.relax, param.beta, param.widthx, param.widthy, param.ubar, param.dt_diag, param.dt_chkpt, param.alpha, param.dt, param.nvisc, param.nkx, param.nky, param.nx, param.ny, param.n_tracers
t_max = param.t_max

# initialize or load data
if init
    println("\nStarting integration from background state.")

    if isfile(dataf)
        error(string("File ", dataf, " already exists.")) 
    else
        const fid = open(dataf, "w")
        println(string("\nWriting data file:    ", dataf))
    end

    # allocate memory for relative vorticity and tracers, their time 
    # tendencies (at 3 time leves), and their forcings
    const i_water = 2
    wtracers = complex(zeros(n_tracers, nkx, nky))
    wtracer_tends = complex(zeros(n_tracers, 3, nkx, nky))
    gforcings = zeros(n_tracers, nx, ny)

    # initialize relative vorticity
    grelvor = gvort_jet
    #= wtracers[1, :, :] = wfft2(grelvor, nkx, nky) =#
    #= wtracers[1, :, :] = zeros(nkx, nky) =#
    #= xx = repeat(x, 1, ny) =#
    #= yy = repeat(y, nx, 1) =#
    #= grelvor = @. exp(-((xx - 4*widthx/9)^2 + yy^2)) - exp(-((xx - 5*widthx/9)^2 + yy^2)) =#
    #= wtracers[1, :, :] = wfft2(grelvor, nkx, nky) =#

    # initialize water vapor
    const qsat = repeat(saturation_fn(param,wavnum, y), nx, 1)
     

    const rel_hum = 0.9
    gwater = rel_hum*qsat
    wtracers[i_water, :, :] = wfft2(gwater, nkx, nky)
     

    # water vapor forcing: evaporation
    gforcings[i_water, :, :] = repeat(evap_fn(param, wavnum, y), nx, 1)
     

    # complete forcing struct
    forcing = Forcing(gvort_jet, gforcings)
    

    # initialize time level indices
    tlev = [1 2 3]
    t = [0]

    model = Model(wavnum, forcing, stirring, wtracers, wtracer_tends, tlev, t)

    # write important parameters and initial state to snapshot file
    write(fid, relax)
    write(fid, beta) 
    write(fid, widthx) 
    write(fid, widthy) 
    write(fid, ubar) 
    write(fid, dt_diag) 
    write(fid, dt_chkpt) 
    write(fid, alpha) 
    write(fid, dt) 
    write(fid, nvisc) 
    write(fid, nkx) 
    write(fid, nky) 
    write(fid, nx) 
    write(fid, ny)
    write(fid, n_tracers)
    cond = zeros(nx, ny)
    snapshot(fid, model, cond)
    t_last_diag = model.t[1]

    #= # write initial checkpoint file =#
    #= t_last_chkpt = t =#
    #= checkpoint(fnbase, t, t_last_diag, t_last_chkpt, wrelvor, wrelvor_tend, tlev) =#
    
    # start integration with one forward step ...
    take_step(model, "single")
    # condense water
    model.wtracers[i_water, :, :], cond = condense(model.wtracers[i_water, :, :])

    # ... and one AB2 step
    take_step(model, "AB2")
    # condense water
    model.wtracers[i_water, :, :], cond = condense(model.wtracers[i_water, :, :])
else
    error("Restart not supported.")
    #= if isfile(string(fnbase, ".h5")) =#
    #=     println(string("\nLoading restart file:    ", fnbase, ".h5")) =#
    #=     t, t_last_diag, t_last_chkpt, wrelvor, wrelvor_tend, tlev = read_checkpoint(fnbase) =#
                    
    #=     fid = open(dataf, "a") =#
    #=     println(string("\nAppending to data file:    ", dataf)) =#
    #= else =#
    #=     error(string("Restart file does not exist:    ", fnbase, ".h5")) =#
    #= end =#
end

# plot 
#= plot_type = "rel. vort." =#
plot_type = "cond"
#= plot_type = "q" =#
#= plot_type = "zonal mean q" =#
#= plot_type = "none" =#

if plot_type == "rel. vort."
    grelvort = gfft2(model.wtracers[1, :, :], nx, ny)
    ph, ax = init_plot(grelvort, "RdBu_r")
elseif plot_type == "cond"    
    ph, ax = init_plot(cond, "viridis")
elseif plot_type == "q"    
    gwater = gfft2(model.wtracers[i_water, :, :], nx, ny)
    ph, ax = init_plot(gwater, "viridis")
    ph.set_clim(vmin=0, vmax=0.8)
elseif plot_type == "zonal mean q"    
    fig, ax = subplots(1)
    gwater = gfft2(model.wtracers[i_water, :, :], nx, ny)
    ax.plot(y', qsat[1, :], "k-", label=L"$q_s$")
    ph, = ax.plot(y', zonal_mean(gwater)', "b-", label=L"$q$")
    ax.set_ylim([0, 1])
    ax.set_xlabel(L"$y$")
    ax.set_ylabel(L"$q$")
    ax.legend(loc="upper right")
end

# begin integration
println(@sprintf("\nCurrent parameter settings:"))
println(@sprintf("\tbeta  = %6.3f", beta))
println(@sprintf("\trelax = %10.3e", relax))
println(@sprintf("\tdt    = %10.3e", dt))
println(@sprintf("\tt_max = %6.1f", t_max))
println(@sprintf("\nStarting integration..."))
while model.t[1] < t_max
    global t_last_diag

    # take a time step
    take_step(model, "AB3")
    # condense water
    model.wtracers[i_water, :, :], cond = condense(model.wtracers[i_water, :, :])
    @info norm(model.wtracers)
    @assert(norm(model.wtracers) ≈ 17244.85817005311)
    error("successful")
    
    if (model.t[1] - t_last_diag) >= dt_diag - sqrt(eps())
        if any(isnan.(model.wtracers))
            error("Floating point exception.")
        end

        println(@sprintf("t = %9.3f", model.t[1]))
        snapshot(fid, model, cond)
        #= println(@sprintf("avg cond rate: %1.3e", mean(cond)/dt)) =#
        t_last_diag = model.t[1]

        if plot_type == "rel. vort."
            grelvort = gfft2(model.wtracers[1, :, :], nx, ny)
            ph.set_array(grelvort[1:end-1, 1:end-1][:])
            vmax = maximum(abs.(grelvort))
            ph.set_clim(vmin=-vmax, vmax=vmax)
            ax.set_title(@sprintf("%s at t=%3.2f", plot_type, model.t[1]))
        elseif plot_type == "cond"
            vmax = maximum(cond)
            cond[cond .== 0.0] .= NaN
            ph.set_array(cond[1:end-1, 1:end-1][:])
            ph.set_clim(vmin=0, vmax=vmax)
            ax.set_title(@sprintf("%s at t=%3.2f", plot_type, model.t[1]))
        elseif plot_type == "q"
            gwater = gfft2(model.wtracers[i_water, :, :], nx, ny)
            ph.set_array(gwater[1:end-1, 1:end-1][:])
            ax.set_title(@sprintf("%s at t=%3.2f", plot_type, model.t[1]))
        elseif plot_type == "zonal mean q"
            gwater = gfft2(model.wtracers[i_water, :, :], nx, ny)
            ph.set_data(y', zonal_mean(gwater)')
            ax.set_title(@sprintf("%s at t=%3.2f", plot_type, model.t[1]))
        end

        #= # CFL =#
        #= u, v = velocities(model) =#
        #= dt_max = min(widthx/(nx - 1)/maximum(abs.(u)), =# 
        #=              widthy/(ny - 1)/maximum(abs.(v))) =#
        #= #1= println(@sprintf("dt_max = %1.3e", dt_max)) =1# =#
        #= println(@sprintf("U_max = %1.3e", max(maximum(abs.(u)), maximum(abs.(v))))) =#

        #= figure() =#
        #= pcolormesh(x[:, 1], y[1, :], u') =#
        #= colorbar() =#
        #= error() =#
    end
    
    if plt.get_fignums() != [1] && plot_type != "none"
        # plot closed, exit
        break
    end

    #= if (t - t_last_chkpt) >= dt_chkpt - sqrt(eps()) =#
    #=     t_last_chkpt = t =#
    #=     chkptf = @sprintf("%s_t_%.1f", fnbase, t) =#
    #=     checkpoint(chkptf, t, t_last_diag, t_last_chkpt, wrelvor, wrelvor_tend, tlev) =#
    #= end =#
end
close(fid)

#= # save restart file =#
#= checkpoint(fnbase, t, t_last_diag, t_last_chkpt, wrelvor, wrelvor_tend, tlev) =#

################################################################################
# end
################################################################################
