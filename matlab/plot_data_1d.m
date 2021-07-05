function [] = plot_data(beta, Gamma, relax, output_fdir)
    nplot = 20;
    dt = 0.5;
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
    fwidth    = 1500; % size of figure 
    fheight   = 480; % height of figure
    fnsize    = 12;
    lwidth    = .4;  % linewidth in points

    fpos      = get(gcf, 'Position');
    fpos      = [0 0 fwidth fheight];
    set(gcf, 'Position', fpos, ...
         'PaperPositionMode', 'auto', ...
         'DefaultLineLineWidth', lwidth, ...
         'visible', 'off') 
    set(0, 'DefaultAxesFontSize', fnsize);

    filename = [fdir fnam '-1d.png'];
    del = 0.5; % time between animation frames

    subplot('Position', [0.04 0.110 0.12 0.815]) 
    plot(mean(data_closure(:,:,600:end),3), y);
    axis([-0.3 0.3 min(y) max(y)])
    xlabel('$$\langle \mathbf{v} \cdot \nabla q \rangle$$','Interpreter','latex')
    ylabel('y')
    grid
    subplot('Position', [0.2 0.110 0.12 0.815]) 
    plot(mean(data_closure_cons(:,:,600:end),3), y);
    axis([-0.1 0.1 min(y) max(y)])
    xlabel('$$\langle v q \rangle $$','Interpreter','latex')
    grid
    subplot('Position', [0.36 0.110 0.12 0.815]) 
    plot(mean(data_u(:,:,600:end),3), y);
    axis([-10 10 min(y) max(y)])
    xlabel('$$\langle u \rangle $$','Interpreter','latex')
    grid
    subplot('Position', [0.52 0.110 0.12 0.815]) 
    plot(mean(data_q(:,:,600:end),3), y);
    axis([-2 2 min(y) max(y)])
    xlabel('$$\langle q \rangle$$','Interpreter','latex')
    grid
    subplot('Position', [0.68 0.110 0.12 0.815])
    plot(mean(data_dq_dy(:,:,600:end),3), y);
    axis([-5 5 min(y) max(y)])
    xlabel('$$\partial_y \langle q \rangle$$','Interpreter','latex')
    grid
    subplot('Position', [0.84 0.110 0.12 0.815]) 
    plot(mean(data_closure_cons(:,:,600:end),3)./mean(data_dq_dy(:,:,600:end),3), y);
    axis([-0.3 0.3 min(y) max(y)])
    xlabel('$$\langle vq \rangle / \partial_y \langle q \rangle$$','Interpreter','latex')
    grid  
    set(findall(gcf,'-property','FontSize'),'FontSize',18);
    saveas(gcf,filename)
    close gcf;
    clear gcf;
end
