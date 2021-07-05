function [] = plot_data(beta, Gamma, relax, output_fdir)

    % read parameters
    params;
    pars.beta = str2double(beta);
    pars.Gamma = str2double(Gamma);
    pars.relax = str2double(relax);
    pars.fdir = [char(output_fdir),'/'];
    fdir = pars.fdir;

    % data directory and file
    fnam      = ['beta=', num2str(pars.beta), '_relax=', num2str(pars.relax)];
    restf     = [fdir, fnam, '.mat'];

    % read parameter and restart file
    load(restf, '-mat')

    % grid axes
    y         = linspace(-pars.widthy/2, pars.widthy/2, pars.ny);

    load([fdir 'data_u.mat']);
    load([fdir 'data_v.mat']);
    load([fdir 'data_q.mat']);
    load([fdir 'data_dq_dy.mat']);
    load([fdir 'data_dq2_dy2.mat']);
    load([fdir 'data_closure.mat']);
    load([fdir 'data_closure_cons.mat']);
    load([fdir 'data_psi.mat']);

    fig = figure('visible', 'off');
    set(gcf, 'Units', 'points', 'visible', 'off')
    fwidth    = 1600; % size of figure 
    fheight   = 1000; % height of figure
    fnsize    = 12;
    lwidth    = .4;  % linewidth in points

    fpos      = get(gcf, 'Position');
    fpos      = [0 0 fwidth fheight];
    set(gcf, 'Position', fpos, ...
         'PaperPositionMode', 'auto', ...
         'DefaultLineLineWidth', lwidth, ...
         'visible', 'off') 
    set(0, 'DefaultAxesFontSize', fnsize);

    filename_w = [fdir fnam '-closure_w.png'];
    filename_q = [fdir fnam '-closure_q.png'];

    % indices of points in the "interior" of the domain
    int_pts   = find(abs(y) < .9 * max(y));
    
    subplot('Position', [0.08 0.08 0.3 0.4]) 
    X = mean(data_q(:,int_pts,600:end), 3)' - pars.beta .* y(int_pts)';
    Y = (mean(data_dq_dy(:,int_pts,600:end), 3)' - pars.beta);
    Z = mean(data_closure(:,int_pts,600:end), 3)';
    f = fit([X, Y], Z, 'linearinterp');
    s= plot(f, [X, Y], Z);
    alpha(s,'z');
    xlabel('$$\omega$$','Interpreter','latex')
    ylabel('$$\partial_y \omega$$','Interpreter','latex')
    zlabel('$$\langle \mathbf{v} \cdot \nabla \omega \rangle$$','Interpreter','latex');
    v = [-8 -2 2];
    [caz,cel] = view(v);
    
    subplot('Position', [0.47 0.08 0.2 0.4])
    f = fit(X, Z, 'linearinterp');
    plot(X, Z, 'o');
    xlabel('$$\omega$$','Interpreter','latex')
    ylabel('$$\langle \mathbf{v} \cdot \nabla \omega \rangle$$','Interpreter','latex');
    
    subplot('Position', [0.75 0.08 0.2 0.4])
    f = fit(Y, Z, 'linearinterp');
    plot(Y, Z, 'o');
    xlabel('$$\partial_y \omega$$','Interpreter','latex')
    ylabel('$$\langle \mathbf{v} \cdot \nabla \omega \rangle$$','Interpreter','latex');

    subplot('Position', [0.08 0.58 0.3 0.4]) 
    X = mean(data_q(:,int_pts,600:end), 3)' - pars.beta .* y(int_pts)';
    Y = (mean(data_dq_dy(:,int_pts,600:end), 3)' - pars.beta);
    Z = mean(data_closure_cons(:,int_pts,600:end), 3)';
    f = fit([X, Y], Z, 'linearinterp');
    s= plot(f, [X, Y], Z);
    alpha(s,'z');
    xlabel('$$\omega$$','Interpreter','latex')
    ylabel('$$\partial_y \omega$$','Interpreter','latex')
    zlabel('$$\langle v \omega \rangle$$','Interpreter','latex');
    v = [-8 -2 2];
    [caz,cel] = view(v);
    
    subplot('Position', [0.47 0.58 0.2 0.4])
    f = fit(X, Z, 'linearinterp');
    plot(X, Z, 'o');
    xlabel('$$\omega$$','Interpreter','latex')
    ylabel('$$\langle v \omega \rangle$$','Interpreter','latex');
    
    subplot('Position', [0.75 0.58 0.2 0.4])
    f = fit(Y, Z, 'linearinterp');
    plot(Y, Z, 'o');
    xlabel('$$\partial_y \omega$$','Interpreter','latex')
    ylabel('$$\langle v \omega \rangle$$','Interpreter','latex');
    
    set(findall(gcf,'-property','FontSize'),'FontSize',18);
    saveas(gcf,filename_w)
    
    subplot('Position', [0.08 0.08 0.3 0.4]) 
    X = mean(data_q(:,int_pts,600:end), 3)';
    Y = (mean(data_dq_dy(:,int_pts,600:end), 3)');
    Z = mean(data_closure(:,int_pts,600:end), 3)';
    f = fit([X, Y], Z, 'linearinterp');
    s= plot(f, [X, Y], Z);
    alpha(s,'z');
    xlabel('$$q$$','Interpreter','latex')
    ylabel('$$\partial_y q$$','Interpreter','latex')
    zlabel('$$\langle \mathbf{v} \cdot \nabla q \rangle$$','Interpreter','latex');
    v = [-8 -2 2];
    [caz,cel] = view(v);
    
    subplot('Position', [0.47 0.08 0.2 0.4])
    f = fit(X, Z, 'linearinterp');
    plot(X, Z, 'o');
    xlabel('$$q$$','Interpreter','latex')
    ylabel('$$\langle \mathbf{v} \cdot \nabla q \rangle$$','Interpreter','latex');
    
    subplot('Position', [0.75 0.08 0.2 0.4])
    f = fit(Y, Z, 'linearinterp');
    plot(Y, Z, 'o');
    xlabel('$$\partial_y q$$','Interpreter','latex')
    ylabel('$$\langle \mathbf{v} \cdot \nabla q \rangle$$','Interpreter','latex');

    subplot('Position', [0.08 0.58 0.3 0.4]) 
    X = mean(data_q(:,int_pts,600:end), 3)';
    Y = (mean(data_dq_dy(:,int_pts,600:end), 3)');
    Z = mean(data_closure_cons(:,int_pts,600:end), 3)';
    f = fit([X, Y], Z, 'linearinterp');
    s= plot(f, [X, Y], Z);
    alpha(s,'z');
    xlabel('$$q$$','Interpreter','latex')
    ylabel('$$\partial_y q$$','Interpreter','latex')
    zlabel('$$\langle v q \rangle$$','Interpreter','latex');
    v = [-8 -2 2];
    [caz,cel] = view(v);
    
    subplot('Position', [0.47 0.58 0.2 0.4])
    f = fit(X, Z, 'linearinterp');
    plot(X, Z, 'o');
    xlabel('$$q$$','Interpreter','latex')
    ylabel('$$\langle v q \rangle$$','Interpreter','latex');
    
    subplot('Position', [0.75 0.58 0.2 0.4])
    f = fit(Y, Z, 'linearinterp');
    plot(Y, Z, 'o');
    xlabel('$$\partial_y q$$','Interpreter','latex')
    ylabel('$$\langle v q \rangle$$','Interpreter','latex');
    
    set(findall(gcf,'-property','FontSize'),'FontSize',18);  
    saveas(gcf,filename_q)

    close gcf;
    clear gcf;
end
