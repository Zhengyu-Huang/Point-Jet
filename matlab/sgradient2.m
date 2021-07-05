function [df_dx, df_dy] = sgradient2(f, kx, ky)
%SGRADIENT  Gradient of grid field
%    [dF_dx, dF_dy] = SGRADIENT(F, kx, ky) evaluates the derivatives
%    of the grid field F with respect to x and y using the spectral
%    transform method. The vectors kx and ky are vectors of
%    wavenumbers.
  
  [nx, ny] = size(f);
  nkx      = length(kx);
  nky      = length(ky);
  
  % assemble wavenumber matrices of same size as wavenumber fields
  % and multiply by i
  ikx        = (i*repmat(kx(:), 1, nky)).^2;
  iky        = (i*repmat(ky(:)', nkx, 1)).^2;
  
  % evaluate derivatives
  wf         = wfft2(f, nkx, nky);
  df_dx      = gfft2(ikx.*wf, nx, ny);
  df_dy      = gfft2(iky.*wf, nx, ny);
