################################################################################
# UTILITY FUNCTIONS FOR 2D POINT JET MODEL
################################################################################
using FFTW, HDF5, PyPlot, Printf

include("struct_defs.jl")



function wfft2(gfield, nkx, nky)
#WFFT2   Grid-to-wave Fourier transform 
#    WFIELD = WFFT2(GFIELD, nkx, nky) transforms the field GFIELD in
#    grid space to the field WFIELD in wavenumber space. The field
#    WFIELD in wavenumber space has size [nkx, nky], which corresponds
#    to retaining the x-Fourier components -(nkx-1),..., (nkx-1) and
#    the y-Fourier components -floor(nky/2),..., ceil(nky/2)-1. If
#    the field GFIELD in grid space has size [nx, ny], the size
#    [nkx, nky] of the field WFIELD in wavenumber space must satisfy
#    nkx <= nx/2 and nky <= ny.
#
#    The inverse of WFFT2 is GFFT2.
  
    nx, ny = size(gfield)
    
    # transform from grid space to wavenumber space
    wfieldl = fft(gfield)
    
    # wavenumber truncation and elimination of redundant coefficients
    kys = Int64.(vcat(1:ceil(nky/2), ny-floor(nky/2)+1:ny))
    wfield = wfieldl[1:nkx, kys]
    
    #= # wavenumber truncation and elimination of redundant coefficients =#
    #= kys = Int64.(vcat(1:ceil(nky/2), ny-floor(nky/2)+1:ny)) =#
    #= wfield = wfieldl[1:nkx, kys] =#
    #= wfield[:, 2:end] = im*imag(wfield[:, 2:end]) =#
    
    return wfield
end

function gfft2(wfield, nx, ny)
#GFFT2   Wave-to-grid Fourier transform. 
#    GFIELD = GFFT2(WFIELD, nx, ny) transforms the field WFIELD in
#    wavenumber space to the field GFIELD on a grid of size [nx, ny].
#
#    GFFT2 is the inverse of WFFT2.
  
    nkx, nky = size(wfield)
    
    # y-indices of nonzero entries in full wavenumber field of size [nx,ny]
    kys = Int64.(vcat(1:ceil(nky/2), ny-floor(nky/2)+1:ny))
    ikys = Int64.(vcat(1, ny:-1:ny-floor(nky/2)+1, ceil(nky/2):-1:2))
    # assemble full wavenumber field (including zero pads)
    wfieldl = complex(zeros(nx, ny))
    wfieldl[1:nkx, kys] = wfield 
    wfieldl[nx:-1:nx-nkx+2, ikys] = conj(wfield[2:nkx, :]) 
    
    # transform from wavenumber space to grid space
    gfield = real(ifft(wfieldl))
    
    return gfield
end

"""
    gfield_filtered = filter(wn, gfield)

Filter a physical field by transforming it to wavenumber space and back.
"""
function filter(wn, gfield)
    nky, ny = length(wn.ky), length(wn.y)
    # spectral filtering 
    filter = 1 .- abs.(wn.ky/maximum(wn.ky))
    #= filter = ones(size(wn.ky)) =#
    #= filter[abs.(wn.ky) .>= 2] .= 0 =#
    wfield = wfft2(gfield, 1, nky)
    gfield = gfft2(filter.*wfield, 1, ny)
    return gfield
end

function forcing_init(param, wn, type = "point_jet", θ = nothing)
    #FORCING_INIT   Initialize forcing.
    x, y = wn.x, wn.y
    nx, ny = length(wn.x), length(wn.y)
    jet_width, widthy = param.jet_width, param.widthy
    # vorticity of jet 
    gvort_jet = zeros(1, ny)
    kx, ky = wn.kx, wn.ky
    nkx, nky = length(kx), length(ky)
    beta = param.beta
    if type == "point_jet"
        
        # point jet:    
        gvort_jet[2:Int64(ny/2-1)] .=  1  
        gvort_jet[Int64(ny/2+1):ny-1] .= -1  
        filter_mask = 1 .- abs.(ky/maximum(ky))  
        wfrcd_trcr = wfft2(gvort_jet, 1, nky)  
        gvort_jet = gfft2(filter_mask.*wfrcd_trcr, 1, ny) 
        #1= plot(y[1, :], gvort_jet[1, :], "b-", label="point jet") =1# =#
    elseif type == "gaussian_jet"
        # gaussian jet:
        # todo bug 
        # gvort_jet[1, :] = @. -(y - widthy/2)*exp(-(y - widthy/2)^2/jet_width^2/2) 
        
        gvort_jet[1, :] = @. -(y)/jet_width^2*exp(-(y)^2/jet_width^2/2) 
        gvort_jet[1, [1, ny]] .= 0  
        # spectral filtering 
        gvort_jet = filter(wn, gvort_jet)

        # coeff = beta*jet_width  
        # gvort_jet *= coeff  
        # stability = @. beta - coeff*exp(-(y - widthy/2)^2/jet_width^2/2)*(1 - (y - widthy/2)^2/jet_width^2/2)  
        # spectral filtering
        # filter_mask = 1 .- abs.(ky/maximum(ky)) 
        # wvort_jet = wfft2(gvort_jet, 1, nky)  
        # gvort_jet = gfft2(filter_mask.*wvort_jet, 1, ny)  
        
    elseif type == "2_gaussian_jets"
        # 2 gaussian jets:
        gvort_jet[1, :] = @. -(y + widthy/4)/jet_width^2*exp(-(y + widthy/4)^2/jet_width^2/2) -
        (y - widthy/4)/jet_width^2*exp(-(y - widthy/4)^2/jet_width^2/2)
        # spectral filtering 
        gvort_jet = filter(wn, gvort_jet)

    elseif type == "customer"
        # 2 gaussian jets:
        gvort_jet[1, :] = @. -(y - θ[1])/jet_width^2*exp(-(y - θ[1])^2/jet_width^2/2) -
        (y - θ[2])/jet_width^2*exp(-(y - θ[2])^2/jet_width^2/2)
        # spectral filtering 
        gvort_jet = filter(wn, gvort_jet)
        
    else
        @error("Forcing type : ", type, " is not recognized.")
    end
    
    #= # plot =#
    #= plot(y[1, :], gvort_jet[1, :], "k-", label="gaussian jet") =#
    #= plot(y[1, :], stability[1, :], "k--", label=L"$\beta - u_{yy}$") =#
    #= axhline(0, lw=1.0, c="k") =#
    #= axvline(0, lw=1.0, c="k") =#
    #= legend() =#
    #= error() =#
    
    # make it a 2D array
    gvort_jet = repeat(gvort_jet, nx, 1)
    
    return gvort_jet
end

function snapshot(fid, m::Model)
#SNAPSHOT  Write model state to file.
#    SNAPSHOT(fid, t, wrelvor) writes a time stamp t and the (complex)
#    tracer field wrelvor in spectral representation to the file with
#    identifier fid.
  
    write(fid, m.t[1])
    write(fid, m.wtracers)
end

function load_pars(fid)
    # parameters at the head of snapshot file
    relax = read(fid, Float64)
    beta = read(fid, Float64) 
    widthx = read(fid, Float64) 
    widthy = read(fid, Float64) 
    ubar = read(fid, Float64) 
    dt_diag = read(fid, Float64) 
    dt_chkpt = read(fid, Float64) 
    dt = read(fid, Float64) 
    nvisc = read(fid, Float64) 
    nkx = read(fid, Int64) 
    nky = read(fid, Int64) 
    nx = read(fid, Int64) 
    ny = read(fid, Int64)
    n_tracers = read(fid, Int64)
    return relax, beta, widthx, widthy, ubar, dt_diag, dt_chkpt, dt, nvisc, nkx, nky, nx, ny, n_tracers
end

function read_snapshot(fid, nkx, nky)
#READ_SNAPSHOT  Read snapshots of model state.
#    [t, wrelvor]=READ_SNAPSHOT(fid, nkx, nky) reads the next time stamp t
#    and the (complex) tracer field wrelvor of size [nkx, nky] from the
#    file with identifier fid.
  
    t = read(fid, Float64)

    wtracers = complex(zeros(1, nkx, nky))
    wtracers = read!(fid, wtracers)

    return t, wtracers
end  
  




function get_wtracer_tends(m::Model)
#DC_DT   Time-tendency of tracer in barotropic or SQG flow.
#      DC_DT(q, c, wavnum, pars) returns the time tendency
#     
#                   dc/dt = -J(Psi, c) + Forcing
#
#      of the tracer c in Fourier space. 
  
    

    nx, ny = m.param.nx, m.param.ny
    nkx, nky = m.param.nkx, m.param.nky
    beta, ubar = m.param.beta, m.param.ubar

    # streamfunction (relvor is i=1)
    wpsi = -m.wavnum.kalpha.*m.wtracers[1, :, :]

    # loop through all the tracers
    
        # advection v ⋅ ∇ω
        wadv = wjacobian(m.wtracers[1, :, :], wpsi, m.wavnum.kx, m.wavnum.ky, nx, ny)
        
        # todo what is the ubar term
        
        # relvor gets a beta*dpsi/dx term
        wadv += -complex.(0, repeat(m.wavnum.kx[:], 1, nky)).*(beta*wpsi + ubar*m.wtracers[1, :, :])
        

        # forcing
        wforcing = wfft2(m.forcing.gforcings[1, :, :], nkx, nky)

        # time tendency of tracer field
        m.wtracer_tends[1, m.tlev[1], :, :] = wadv + wforcing

end

function wjacobian(wA, wB, kx, ky, nx, ny)
#WJACOBIAN  Jacobian in wavenumber space
#    WJ = WJACOBIAN(WA, WB, kx, ky, nx, ny) evaluates the Jacobian WJ
#    of the fields WA and WB in wavenumber space using the spectral
#    transform method. The vectors kx and ky are vectors of
#    wavenumbers, and the scalars nx and ny give the size of the
#    transform grid on which the products are evaluated.
  
    if size(wB) != size(wA)
        error("Input fields must be of equal size.")
    end
    nkx, nky = size(wA)
    
    # assemble wavenumber matrices of same size as wavenumber fields
    # and multiply by i
    ikx = im*repeat(kx[:], 1, nky)
    iky = im*repeat(ky[:]', nkx, 1)
    
    # evaluate Jacobian
    gC = gfft2(ikx.*wA, nx, ny).*gfft2(iky.*wB, nx, ny) - gfft2(ikx.*wB, nx, ny).*gfft2(iky.*wA, nx, ny)
    wC = wfft2(gC, nkx, nky)

    return wC
end

function stirring_init(param::Param, wn)
#WSTIRRING_INIT   Initialize stochastic stirring forcing.
    dt, jet_width = param.dt, param.jet_width
    widthx, widthy = param.widthx, param.widthy
    x, y = wn.x, wn.y
    nx, ny = length(x), length(y)
    nkx, nky = length(wn.kx), length(wn.ky)
    # decorrelation time 
    decorr_time = 4.0 # 2 days since t = 1 corresponds to 12 hrs

    # coefficients in finite difference scheme
    a = sqrt(1 - exp(-2*dt/decorr_time))
    b = exp(-dt/decorr_time)

    # only force wavenumbers 10 < sqrt(kx^2 + ky^2) < 14 and kx > 3
    kx_grid = repeat(wn.kx, 1, nky)
    kxy2 = repeat(wn.kx.^2, 1, nky) + repeat(wn.ky.^2, nkx, 1)
    wavnums_mask = ones(nkx, nky)
    wavnums_mask[kxy2 .< 10^2] .= 0
    wavnums_mask[kxy2 .> 14^2] .= 0
    wavnums_mask[kx_grid .< 3] .= 0

    # variance of brownian motion 
    #= variance = decorr_time/2 =#
    #= variance = 500 =#
    #= variance = 1e2 =#
    variance = 1e1

    # confine the stirring to the center of the domain
    #= stir_width = p.widthy/4 =#
    #= location = repeat(exp.(-(y .- p.widthy/2).^2/stir_width^2/2), p.nx, 1) =#
    #= stir_width = p.widthy/8 =#
    stir_width = jet_width
    location = repeat(exp.(-(y .+ widthy/4).^2/stir_width^2/2) .+ exp.(-(y .- widthy/4).^2/stir_width^2/2), nx, 1)

    # initial stirring 
    wstir = variance*randn(nkx, nky)
    wstir .*= wavnums_mask
    gstir = gfft2(wstir, nx, ny)
    gstir .*= location
    wstir = wfft2(gstir, nkx, nky)

    #= pcolormesh(x[:, 1], y[1, :], gstir', cmap="bwr") =#
    #= colorbar() =#
    #= error() =#

    # save as struct
    stirring = Stirring(decorr_time, a, b, wavnums_mask, variance, location, wstir)

    return stirring
end

function update_wstir(m::Model)
#WSTIRRING  Stochastic stirring forcing in wavenumber space
#    wstir = WSTIRRING(p, sp, wstir) evaluates stochastic stirring forcing 
#    for next time step given 
#       p:     the parameter struct
#       sp:    the stirring parameter struct
#       wstir: the stirring forcing from the previous timestep in wavenumber
#              space
    
    nkx, nky = m.param.nkx, m.param.nky
    nx, ny = m.param.nx, m.param.ny

    m.stirring.wstir[:, :] = m.stirring.a*m.stirring.variance*randn(nkx, nky) + m.stirring.b*m.stirring.wstir
    m.stirring.wstir[:, :] .*= m.stirring.wavnums_mask
    gstir = gfft2(m.stirring.wstir, nx, ny)
    gstir .*= m.stirring.location
    m.stirring.wstir[:, :] = wfft2(gstir, nkx, nky)
end

function step_ab2t(y, F, tlev, damping, dt)
#STEP_AB2T   Second-order Adams-Bashforth-trapezoidal time step.
#    STEP_AB2T(y, F, tlev, damping, dt) performs a second order
#    Adams-Bashcroft time step to advance the solution y of the
#    ordinary differential equation
#         
#                    dy/dt = F - damping .* y
#
#    by one time step. The damping coefficient must be either a
#    constant scalar or a constant matrix of the same size as y. The
#    right hand side F must be given at two time levels: F{tlev(1)} at
#    the current time level n, at which the solution y is given on
#    input and F{tlev(2)} at the previous time level n-1. That is,
#    the right hand side is assumed to be given as a cell 2-vector F
#    of right-hand sides for two time levels, with the index vector
#    tlev indicating which element in the cell corresponds to which
#    time level.
#   
#    The right-hand side F is differenced by a second-order
#    Adams-Bashcroft difference the damping term is differenced by a
#    semi-implicit trapezoidal difference.
  
    return @. (y + dt/2*(3*F[tlev[1], :, :] - F[tlev[2], :, :]) - 0.5*dt*damping*y)/(1 + 0.5*dt*damping)
end

function  step_ab3t(y, F, tlev, damping, dt)
#STEP_AB3T   Third-order Adams-Bashforth-trapezoidal time step.
#    STEP_AB3T(y, F, tlev, damping, dt) performs a third order
#    Adams-Bashforth time step to advance the solution y of the
#    ordinary differential equation
#         
#                    dy/dt = F - damping .* y
#
#    by one time step. The damping coefficient must be either a
#    constant scalar or a constant matrix of the same size as y. The
#    right hand side F must be given at three time levels: F{tlev(1)}
#    at the current time level n, at which the solution y is given on
#    input F{tlev(2)} at the time level n-1 one step back and
#    F{tlev(3)} at the time level n-2 two steps back. That is, the
#    right hand side is assumed to be given as a cell 3-vector F of
#    right-hand sides for different time levels, with the index vector
#    tlev indicating which element in the cell corresponds to which
#    time level.
#   
#    The right-hand side F is differenced by a third-order
#    Adams-Bashcroft difference the damping term is differenced by a
#    semi-implicit trapezoidal difference.

#    Reference: D. Durran, Numerical Methods for Wave Equations in
#    Geophysical Fluid Dynamics, Springer (1999), p. 143.
  
    return @. (y + dt/12*(23*F[tlev[1], :, :] - 16*F[tlev[2], :, :] + 5*F[tlev[3], :, :]) - 0.5*dt*damping*y)./(1 + 0.5*dt*damping)
end





"""
    take_step(m, step_type)

Step model `m` forward in time using the method `step_type`.
"""
function take_step(m::Model, step_type)
    # todo why dt_factor = 2/3
    if step_type == "single"
        stepper = step_ab2t
        dt_factor = 2/3
    elseif step_type == "AB2"
        stepper = step_ab2t
        dt_factor = 1
    elseif step_type == "AB3"
        stepper = step_ab3t
        dt_factor = 1
    end
    nx, ny, dt = m.param.nx, m.param.ny, m.param.dt
    relax = m.param.relax

    # update relvor forcing
    m.forcing.gforcings[1, :, :] = relax*(m.forcing.gvort_jet - gfft2(m.wtracers[1, :, :], nx, ny))
    
    # compute tendencies
    get_wtracer_tends(m)

    # add stirring to break the symmetry and lead to chaos 
    update_wstir(m)
    m.wtracer_tends[1, m.tlev[1], :, :] += m.stirring.wstir

    # take step
    
            # use hyperdiffusion for relvor
            m.wtracers[1, :, :] = stepper(m.wtracers[1, :, :], 
                                          m.wtracer_tends[1, :, :, :], m.tlev, 
                                          m.wavnum.hyperdiff, 
                                          dt_factor*dt)



    # update tlev and t
    m.tlev[:] = m.tlev[[3 1 2]]
    m.t[:] .+= dt
end




"""
    ph, ax = init_plot(x, y, field, cmap)

Begin a plot of 2D `field` for animations. Uses colormap `cmap` and returns plot handle `ph` and 
axis `ax`.
"""
function init_plot(x, y, field, cmap)
    fig, ax = subplots(1)
    ph = ax.pcolormesh(x[:, 1], y[1, :], field', cmap=cmap)
    colorbar(ph, ax=ax)
    ax.set_title("t=0.0")
    return ph, ax
end

"""
    zm = zonal_mean(field)

Returns 1D array `zm` representing zonal mean (mean in x direction) of 2D `field`.
"""
function zonal_mean(field)
    nx = size(field, 1)
    return sum(field, dims=1)/nx
end




#############################################################################


"""
    u, v = velocities(m)

Computes the x-velocity field `u` and the y-velocity field `v` from relative
vorticity of current model state `m`.
"""
function velocities(m::Model)
    # streamfunction 
    wpsi = -m.wavnum.kalpha.*m.wtracers[1, :, :]
    
    # zonal velocity
    iky = im*repeat(m.wavnum.ky[:]', m.nkx, 1)
    u = -gfft2(iky.*wpsi, m.nx, m.ny) .+ m.ubar
    
    # meridional velocity
    ikx = im*repeat(m.wavnum.kx[:], 1, m.nky)
    v = gfft2(ikx.*wpsi, m.nx, m.ny)
  
    return u, v
end
function velocities(wrelvor, wavnum)
    # streamfunction
    wpsi = -wavnum.kalpha.*wrelvor

    # zonal velocity
    iky = im*repeat(wavnum.ky[:]', nkx, 1)
    u = -gfft2(iky.*wpsi, nx, ny) .+ ubar

    # meridional velocity
    ikx = im*repeat(wavnum.kx[:], 1, nky)
    v = gfft2(ikx.*wpsi, nx, ny)

    return u, v
end