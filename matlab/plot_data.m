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
    load([fdir 'data_w.mat']);
    load([fdir 'data_dw_dy.mat']);
    load([fdir 'data_closure.mat']);
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

    filename = [fdir fnam '.gif'];
    del = 0.5; % time between animation frames

    for i = 1:size(data_u,3) 
        if mod(i, nplot) == 1 || nplot == 1
            subplot('Position', [0.04 0.110 0.12 0.815]) 
            plot(data_closure(:,:,i), y);
            title(['$$\langle v \cdot \nabla \omega \rangle$$ (t=', sprintf('%3.2f', dt*(i-1)), ')'],'Interpreter','latex')
            axis([-0.5 0.5 min(y) max(y)])
            xlabel('$$\langle v \cdot \nabla \omega \rangle$$','Interpreter','latex')
            ylabel('y')
            grid
            subplot('Position', [0.2 0.110 0.12 0.815]) 
            plot(data_u(:,:,i), y);
            title(['$$\langle v_x \rangle $$ (t=', sprintf('%3.2f', dt*(i-1)), ')'],'Interpreter','latex')
            axis([-10 10 min(y) max(y)])
            xlabel('$$\langle v_x \rangle $$','Interpreter','latex')
            grid
            subplot('Position', [0.36 0.110 0.12 0.815]) 
            plot(data_v(:,:,i), y);
            title(['Mean $$\langle v_y \rangle $$ (t=', sprintf('%3.2f', dt*(i-1)), ')'],'Interpreter','latex')
            axis([-1 1 min(y) max(y)])
            xlabel('$$\langle v_y \rangle $$','Interpreter','latex')
            grid
            subplot('Position', [0.52 0.110 0.12 0.815]) 
            plot(data_w(:,:,i), y);
            title(['$$\langle \omega \rangle$$ (t=', sprintf('%3.2f', dt*(i-1)), ')'],'Interpreter','latex')
            axis([-2 2 min(y) max(y)])
            xlabel('$$\langle \omega \rangle$$','Interpreter','latex')
            grid
            subplot('Position', [0.68 0.110 0.12 0.815])
            plot(data_dw_dy(:,:,i), y);
            title(['$$\partial_y \langle \omega \rangle$$ (t=', sprintf('%3.2f', dt*(i-1)), ')'],'Interpreter','latex')
            axis([-5 5 min(y) max(y)])
            xlabel('$$\partial_y \langle \omega \rangle$$','Interpreter','latex')
            grid
            subplot('Position', [0.84 0.110 0.12 0.815]) 
            plot(data_psi(:,:,i), y);
            title(['$$\langle \Psi \rangle$$ (t=', sprintf('%3.2f', dt*(i-1)), ')'],'Interpreter','latex')
            axis([-10 10 min(y) max(y)])
            xlabel('$$\langle \Psi \rangle$$','Interpreter','latex')
            grid  
            frame = getframe(fig); 
            %M(i) = frame; 
            im = frame2im(frame);
            [imind,cm] = rgb2ind(im,256);
            if i == 1
              imwrite(imind,cm,filename,'gif','Loopcount',inf,'DelayTime',del);
            else
              imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',del);
            end
        end
    end
%     % show movie
%     figure(1)
%     clf
%     set(gca, 'Position', [0 0 1 1], ...
%          'Box', 'off', ...
%          'Ticklength', [0 0])
% 
%     movie(M, 4, 3)
    close gcf;
    clear gcf;
end