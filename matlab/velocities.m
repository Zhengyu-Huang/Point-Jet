function [u, v] = velocities(wq, wn, p)%VELOCITIES    Velocities on grid.%    [u, v]=VELOCITIES(wq, wavnum, pars) computes the x-velocity field%    u and the y-velocity field v from a given tracer field wq in%    Fourier space. The structures wavnum and pars contain wavenumber%    and mean-flow/dimension parameters.    % streamfunction   wpsi       = - wn.kalpha .* wq;    % zonal velocity  iky        = i*repmat(wn.ky(:)', p.nkx, 1);  u          = - gfft2(iky.*wpsi, p.nx, p.ny) + p.ubar;    % meridional velocity  ikx        = i*repmat(wn.kx(:), 1, p.nky);  v          = gfft2(ikx.*wpsi, p.nx, p.ny);  