params;
pars.widthx = 2*pi
pars.widthy = 2*pi

% refinement level
N = 0
pars.nkx       = pars.nkx*2^N;
pars.nky       = pars.nky*2^N;
pars.nx        = pars.nx*2^N;
pars.ny        = pars.ny*2^N;

% pars.nkx       = 16;
% pars.nky       = 16;
% pars.nx        = 16*2;
% pars.ny        = 16*2;
% extract dimension parameters
nkx       = pars.nkx;
nky       = pars.nky;
nx        = pars.nx;
ny        = pars.ny;
wavnum    = wavnum_init(pars);

% grid axes
xx         = linspace(0, pars.widthx, nx)';
yy         = linspace(-pars.widthy/2, pars.widthy/2, ny);


w = zeros( nx, ny);
u = zeros( nx, ny);
v = zeros( nx, ny);
dw_dx = zeros( nx, ny);
dw_dy = zeros( nx, ny);
closure = zeros( nx, ny);
psi = zeros( nx, ny);

for i = 1:nx
    for j = 1:ny
        x = xx(i);
        y = yy(j);

        u(i,j) = -cos(x)*sin(y);
        v(i,j) = sin(x)*cos(y);
        w(i,j) = 2*cos(x)*cos(y);
        psi(i,j) = -cos(x)*cos(y);
        
        dw_dx(i,j) = -2*sin(x)*cos(y);
        dw_dy(i,j) = -2*cos(x)*sin(y);
        
        closure(i,j) = u(i,j)*dw_dx(i,j)+v(i,j)*dw_dy(i,j);
    end
end

wq = wfft2(w, nkx, nky);


% test 
w_test = gfft2(wq, nx, ny);
disp('w fft rel. error is '); disp(norm(w - w_test)/norm(w));

psi_test = gfft2(- wavnum.kalpha .* wq, nx, ny);
disp('psi rel. error is '); disp(norm(psi - psi_test)/norm(psi));


[u_test,v_test] = velocities(wq,wavnum,pars);
u_test = u_test - pars.ubar;
disp('u rel. error is '); disp(norm(u - u_test)/norm(u));
disp('v rel. error is '); disp(norm(v - v_test)/norm(v));


gpv_test = pv(wq, pars, y);
w_test = gpv_test - pars.beta*y;
disp('w rel. error is '); disp(norm(w - w_test)/norm(w));

[dw_dx_test, dw_dy_test] = sgradient(w, wavnum.kx, wavnum.ky);
closure_test = u_test.*dw_dx_test+v_test.*dw_dy_test;

disp('dw_dx rel. error is '); disp(norm(dw_dx - dw_dx_test)/norm(dw_dx));
disp('dw_dy rel. error is '); disp(norm(dw_dy - dw_dy_test)/norm(dw_dy));
disp('closure rel. error is '); disp(norm(closure - closure_test, Inf)/nx);

h = pcolor(psi - psi_test)
set(h, 'EdgeColor', 'none');
colorbar()

h = pcolor(u)
set(h, 'EdgeColor', 'none');
colorbar()