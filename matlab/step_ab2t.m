function  y_up = step_ab2t(y, F, tlev, damping, dt)%STEP_AB2T   Second-order Adams-Bashforth-trapezoidal time step.%    STEP_AB2T(y, F, tlev, damping, dt) performs a second order%    Adams-Bashcroft time step to advance the solution y of the%    ordinary differential equation%         %                    dy/dt = F - damping .* y%%    by one time step. The damping coefficient must be either a%    constant scalar or a constant matrix of the same size as y. The%    right hand side F must be given at two time levels: F{tlev(1)} at%    the current time level n, at which the solution y is given on%    input; and F{tlev(2)} at the previous time level n-1. That is,%    the right hand side is assumed to be given as a cell 2-vector F%    of right-hand sides for two time levels, with the index vector%    tlev indicating which element in the cell corresponds to which%    time level.%   %    The right-hand side F is differenced by a second-order%    Adams-Bashcroft difference; the damping term is differenced by a%    semi-implicit trapezoidal difference.    if nargin ~= 5    error('Wrong number of arguments. See STEP_AB2T.')  end    y_up = ( y + ...	   dt/2 * ( 3*F{tlev(1)} - F{tlev(2)} ) ...	   - 0.5 * dt * damping .* y ) ./ ( 1 + 0.5 * dt * damping);