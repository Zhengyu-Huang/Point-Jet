clear all

nplot     = 20;           % plot every nplot-th saved frame
nprint    = 100*nplot;   % save every nprint-th saved frame as PostScript

% read parameters
%params
fdir      = './data/';

% data directory and file
fnam      = 'beta=1_relax=0.16';
dataf     = [fdir, fnam, '.dat'];
restf     = [fdir, fnam, '.mat'];
  
% read parameter and restart file
load(restf, '-mat')

disp(sprintf('\nCurrent parameter settings:'))
disp(sprintf('\tbeta  = %6.3f', pars.beta))
disp(sprintf('\trelax = %10.3e', pars.relax))

% extract dimension parameters
nkx       = pars.nkx;
nky       = pars.nky;
nx        = pars.nx;
ny        = pars.ny;

% grid axes
x         = linspace(0, pars.widthx, nx)';
y         = linspace(-pars.widthy/2, pars.widthy/2, ny);
dy        = gradient(y);

% initialize wavenumbers 
wavnum    = wavnum_init(pars);

% indices of points in the "interior" of the domain
int_pts   = find(abs(y) < .95 * max(y));

nsave     = 0;
fid       = fopen(dataf, 'r', 'ieee-le');
[t, wq]   = read_snapshot(fid, nkx, nky);
warning off
[u,v] = velocities(wq,wavnum,pars);
u = u - pars.ubar;
gpv = pv(wq, pars, y);
w = gpv - pars.beta*y;
[dw_dx, dw_dy] = sgradient(w, wavnum.kx, wavnum.ky);
closure = u.*dw_dx+v.*dw_dy;
psi = gfft2(- wavnum.kalpha .* wq, nx, ny);
data_u = mean(u);
data_v = mean(v);
data_w = mean(w);
data_dw_dy = mean(dw_dy);
data_closure = mean(closure);
data_psi = mean(psi);
while ~feof(fid)
  nsave   = nsave + 1;
  if mod(nsave, nplot) == 0 
    [u,v] = velocities(wq,wavnum,pars);
    u = u - pars.ubar;
    gpv = pv(wq, pars, y);
    w = gpv - pars.beta*y;
    [dw_dx, dw_dy] = sgradient(w, wavnum.kx, wavnum.ky);
    closure = u.*dw_dx+v.*dw_dy;
    psi = gfft2(- wavnum.kalpha .* wq, nx, ny);

    data_u = cat(3,data_u,mean(u));
    data_v = cat(3,data_v,mean(v));
    data_w = cat(3,data_w,mean(w));
    data_dw_dy = cat(3,data_dw_dy,mean(dw_dy));
    data_closure = cat(3,data_closure,mean(closure));
    data_psi = cat(3,data_psi,mean(psi));
  end
  [t, wq] = read_snapshot(fid, nkx, nky);
end
warning on
fclose(fid);
save([fdir 'data_u.mat'],'data_u');
save([fdir 'data_v.mat'],'data_v');
save([fdir 'data_w.mat'],'data_w');
save([fdir 'data_dw_dy.mat'],'data_dw_dy');
save([fdir 'data_closure.mat'],'data_closure');
save([fdir 'data_psi.mat'],'data_psi');