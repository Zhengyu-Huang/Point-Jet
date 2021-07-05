function [] = load_data(beta, Gamma, relax, output_fdir)

    nplot     = 1;           % plot every nplot-th saved frame

    % read parameters
    params;
    pars.beta = str2double(beta);
    pars.Gamma = str2double(Gamma);
    pars.relax = str2double(relax);
    pars.fdir = [char(output_fdir),'/'];
    fdir = pars.fdir;

    % data directory and file
    fnam      = ['beta=', num2str(pars.beta), '_relax=', num2str(pars.relax)];
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
    %[dw2_dxdy, dw2_dy2] = sgradient(dw_dy, wavnum.kx, wavnum.ky);
    [dw2_dx2, dw2_dy2] = sgradient2(w, wavnum.kx, wavnum.ky);
    closure = u.*dw_dx+v.*dw_dy;
    closure_cons = v.*w;
    psi = gfft2(- wavnum.kalpha .* wq, nx, ny);
    data_u = mean(u);
    data_v = mean(v);
    data_q = mean(w) + pars.beta*y;
    data_dq_dy = mean(dw_dy) + pars.beta;
    data_dq2_dy2 = mean(dw2_dy2);
    data_closure = mean(closure);
    data_closure_cons = mean(closure_cons);
    data_psi = mean(psi);
    while ~feof(fid)
      nsave   = nsave + 1;
      if mod(nsave, nplot) == 0 
        [u,v] = velocities(wq,wavnum,pars);
        u = u - pars.ubar;
        gpv = pv(wq, pars, y);
        w = gpv - pars.beta*y;
        [dw_dx, dw_dy] = sgradient(w, wavnum.kx, wavnum.ky);
        %[dw2_dxdy, dw2_dy2] = sgradient(dw_dy, wavnum.kx, wavnum.ky);
        [dw2_dx2, dw2_dy2] = sgradient2(w, wavnum.kx, wavnum.ky);
        closure = u.*dw_dx+v.*dw_dy;
        closure_cons = v.*w;
        psi = gfft2(- wavnum.kalpha .* wq, nx, ny);

        data_u = cat(3,data_u,mean(u));
        data_v = cat(3,data_v,mean(v));
        data_q = cat(3,data_q,mean(w) + pars.beta*y);
        data_dq_dy = cat(3,data_dq_dy,mean(dw_dy) + pars.beta);
        data_dq2_dy2 = cat(3,data_dq2_dy2,mean(dw2_dy2));
        data_closure = cat(3,data_closure,mean(closure));
        data_closure_cons = cat(3,data_closure_cons,mean(closure_cons));
        data_psi = cat(3,data_psi,mean(psi));
      end
      [t, wq] = read_snapshot(fid, nkx, nky);
    end
    warning on
    fclose(fid);
    save([fdir 'data_u.mat'],'data_u');
    save([fdir 'data_v.mat'],'data_v');
    save([fdir 'data_q.mat'],'data_q');
    save([fdir 'data_dq_dy.mat'],'data_dq_dy');
    save([fdir 'data_dq2_dy2.mat'],'data_dq2_dy2');
    save([fdir 'data_closure.mat'],'data_closure');
    save([fdir 'data_closure_cons.mat'],'data_closure_cons');
    save([fdir 'data_psi.mat'],'data_psi');
end
