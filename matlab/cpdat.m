%CPDAT   Copy part of data file into new file.clear allfclose('all');% data directory and filefnam     = 'beta=1_relax=0.000625';fdir     = '/home/tapio/data/ptjet/';dataf    = [fdir, fnam, '.dat'];restf    = [fdir, fnam, '.mat'];ndataf   = ['/work/tapio/', fnam, '.nat'];% read parameter and restart fileload(restf, '-mat')% copy up to time t0t0       = t_last_diag;% extract dimension parametersnkx      = pars.nkx;nky      = pars.nky;fold     = fopen(dataf, 'r', 'ieee-le');fnew     = fopen(ndataf, 'w', 'ieee-le');[t, wq]  = read_snapshot(fold, nkx, nky);while ~feof(fold) & t <= t0+sqrt(eps)  snapshot(fnew, t, wq)  [t, wq] = read_snapshot(fold, nkx, nky);endfclose(fold);fclose(fnew);