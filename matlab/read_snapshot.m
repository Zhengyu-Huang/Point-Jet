function [t, wq] = read_snapshot(fid, nkx, nky)%READ_SNAPSHOT  Read snapshots of model state.%    [t, wq]=READ_SNAPSHOT(fid, nkx, nky) reads the next time stamp t%    and the (complex) tracer field wq of size [nkx, nky] from the%    file with identifier fid.    t     = fread(fid, 1, 'single');  rwq   = fread(fid, [nkx, nky], 'single');  iwq   = fread(fid, [nkx, nky], 'single');  wq    = complex(rwq, iwq);