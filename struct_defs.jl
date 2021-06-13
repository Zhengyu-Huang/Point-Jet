################################################################################
# Define some structs that will be useful for the 2d fluid model
################################################################################

mutable struct Param
    
    # Physics information
    relax::Float64   # Inverse relaxation time scale of forcing
    beta::Float64    # Mean potential vorticity gradient  
    alpha::Float64    # ALPHA: FT(advected variable) = -(K^alpha) * FT(streamfunction)
                      # alpha = 2 for barotropic vorticity equation
                      # alpha = 1 for surface quasi-geostrophy
    n_tracers::Int64
    jet_width::Float64
    ubar::Float64    # Background flow s.t. u0=0 at channel center 
    
    nvisc::Float64    # time step and damping of highest wavenumber

    # global warming parameters
    deltaT::Float64  
    alpha_qsat::Float64  
    alpha_evap::Float64  

    # Grid information
    widthx::Float64  # Channel length
    widthy::Float64  # Channel width
    nkx::Int64        # dimension of fields in wavenumber space
    nky::Int64 

    nx::Int64        # dimension of fields in grid space (large enough to ensure unaliased products)
    ny::Int64 
    # domain (useful for plotting and defining forcings)
    x::Array{Float64, 2}
    y::Array{Float64, 2}
    
    
    # Time info
    t_max::Float64
    dt_diag::Float64   # Time interval for saving tracer field
    dt_chkpt::Float64  # Time interval for checkpointing of simulation

    dt::Float64 

    
    
    # Data directory and file info
    init::Bool   # initialize?
    fdir::String  

    fnam::String  
    fnbase::String 
    dataf::String 

end



function Param(;relax = 0.02, beta = 0.0, alpha = 2.0, n_tracers = 2, 
    ubar = 0.0, nvisc = -1.0,
    #
    deltaT = 0.0, alpha_qsat = 0.07*deltaT, alpha_evap = 0.025*deltaT,
    #
    widthx = 8*pi, widthy = 4*pi, jet_width = widthy/32, nkx = 2^6, nky = 2nkx, 
    #
    t_max = 1000.0,  dt_diag = 0.5, dt_chkpt = 25.0, dt = -1.0, 
    #
    init = true, fdir = "./data/", fnam = string("beta_", beta, "_relax_", relax), 
    fnbase = string(fdir, fnam), dataf = string(fnbase, ".dat")
    )
    
    
    if nvisc < 0.0
        # time step and damping of highest wavenumber
        if nky <= 2^7
            # dt_max for nky = 2^7 is about 0.1 so do 0.05 to be safe
            nvisc = 1.0e3
        elseif nky == 2^8
            nvisc = 5.0e2
            
        else
            error("nvisc is required")
        end
    end
    
    
    if dt < 0.0
        # time step and damping of highest wavenumber
        if nky <= 2^7
            # dt_max for nky = 2^7 is about 0.1 so do 0.05 to be safe
            dt = 5e-2
        elseif nky == 2^8
            dt = 2.5e-2
        else
            error("dt is required")
        end
    end
    
    nx = Int64(round(3/2*maximum([nkx, nky])))
    ny = nx
    x = zeros(nx, 1)
    y = zeros(1, ny)
    x[:, 1] = (0:1.0/(nx-1):1.0)*widthx
    y[1, :] = (-0.5:1.0/(ny-1):0.5)*widthy
    
    return Param(relax, beta, alpha, n_tracers, jet_width, ubar, nvisc,
    deltaT, alpha_qsat, alpha_evap, 
    widthx, widthy, nkx, nky, nx, ny, x, y,
    t_max, dt_diag, dt_chkpt, dt, 
    init, fdir, fnam, fnbase, dataf)
    
    
end







"""
Wavenumbers

This struct holds arrays useful for computations in wavenumber space.
"""
struct Wavenumbers
    nx::Int64
    ny::Int64
    kx::Array{Float64}
    ky::Array{Float64}
    kalpha::Array{Float64}
    hyperdiff::Array{Float64}
    diff::Array{Float64}
end

"""
Forcing

This struct holds arrays of forcings applied to the system.
"""
struct Forcing
    gvort_jet::Array{Float64}
    gforcings::Array{Float64}
end

"""
Stirring

This struct holds the parameters and arrays to calculate stochastic stirring.
"""
struct Stirring
    decorr_time::Float64
    a::Float64
    b::Float64
    wavnums_mask::Array{Int64}
    variance::Float64
    location::Array{Float64}
    wstir::Array{Complex{Float64}}
end

"""
Model

This struct wraps the other structs together and contains arrays describing the 
state of the model at time `t`.
"""
struct Model
    wavnum::Wavenumbers
    forcing::Forcing
    stirring::Stirring
    wtracers::Array{Complex{Float64}}
    wtracer_tends::Array{Complex{Float64}}
    tlev::Array{Int64}
    t::Array{Float64}
end
