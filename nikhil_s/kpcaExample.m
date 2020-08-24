if ispc
    addpath(genpath('C:/Users/nkhld/Dropbox/ARClab/TwoLink/GaussianBeliefSpaceTwoLink'));
    addpath('C:/Users/nkhld/Dropbox/ARClab/TwoLink/mexTests');
    addpath('C:/Users/nkhld/Dropbox/MATLAB/');
    addpath('C:/Users/nkhld/Dropbox/MATLAB/cbrewer/');
else
    addpath(genpath('/home/nikhil/Dropbox/ARClab/TwoLink/GaussianBeliefSpaceTwoLink'));
    addpath('/home/nikhil/Dropbox/ARClab/TwoLink/mexTests');
    addpath('/home/nikhil/Dropbox/MATLAB/');
    addpath('/home/nikhil/Dropbox/MATLAB/cbrewer/');
end
clear; close all; clc

rng(7);
time_delay = 0.05;
d = 7;
loadBaxter;
%%
make_patch = @(o, color) patch('XData', o.XData, 'YData', o.YData, 'ZData', o.ZData, 'edgecolor', 'none', 'visible', 'on', 'facecolor', color);
make_edge_patch = @(o) patch('XData', o.XData, 'YData', o.YData, 'ZData', o.ZData, 'edgecolor', 'k', 'visible', 'on', 'facecolor', 'none');

mesh_mask = zeros(1,2*d); mesh_mask([1,5,9,13]) = 1;
fk_mask = right.a | right.d;
%%
F = [1 2 6 5; 2 4 8 6; 3 4 8 7; 1 3 7 5; 1 2 4 3; 5 6 8 7]';
obs_template = 0.12*[-1    -1    -1; 1    -1    -1;    -1     1    -1; 1     1    -1; -1    -1     1; 1    -1     1;  -1     1     1;    1     1     1];
set(gca, 'Projection', 'perspective');
%%
% set up obstacles
numObs = 3; offset = [0.5 -0.25 0];
offset(2:numObs,:) = [rand(numObs-1, 1), 2*rand(numObs-1, 2) - 1];
% offset = [0.8*rand -0.8*rand 0.8*rand-0.4];

% c = parula(numObs);
colors = cbrewer('qual', 'Set1', numObs + 1);
for i = numObs:-1:1
    obs(i) = V2shapeMex(bsxfun(@plus, obs_template, offset(i,:)), F);
    obs_h(i) = make_patch(obs(i), colors(i+1,:));
end
%%
N = 5000;

data = 2*rand(N,d) - 1;
data_u = unscale(data);
for i = 1:numObs
    y(:,i) = i/2*(1-GJKarray(generateAllArmPolyhedra(right,data_u,w*1,mesh_mask),obs(i),6));
end
y = max(y,[],2);
%%
savePlot = false;
for k = [1 2 3]
    switch k
        case 1
            g = 1; % fk
            Kfun = @(x, y) robotKernel3D(right, x, y, g, fk_mask);
            kernelType = 'FK';
        case 2
            g = 0.01; % gaussian
            Kfun = @(x, y) exp(-g*pdist2(x, y).^2);
            kernelType = 'Gauss';
            data_u = data;
        case 3
            g = 0.01; % rat quad
            Kfun = @(x, y) 1./(1+g/2*pdist2(x, y).^2).^2;
            kernelType = 'RQ';
            data_u = data;
        case 4
            Kfun = @(x, y) (x*y'+1).^2;
            kernelType = 'Polynomial';
        case 5
            Kfun = @(x, y) tanh(x*y');
            kernelType = 'tanh';
        case 6
            g = 1;
            numPoints = 5;
            o = generateRobotPointOffsets(R,w,numPoints);
            Kfun = @(x,y) robotKernelMultipoint(R,x,y,g,o);
            kernelType = 'FK Multi';
    end
    K = Kfun(data_u, data_u);
    
    oneN = 1/N*ones(N);
    K = K - oneN*K - K*oneN + oneN*K*oneN;
    %%
    [a, eigvalues] = eigs(K,3);
    [eigvalues, idx] = sort(diag(eigvalues), 'descend');
    
    a = a./sqrt(abs(eigvalues'));
    a = a(:, idx);
    %%
    Nt = 2000;
    qt = pi*(2*rand(Nt,d) - 1);
    qt_u = unscale(qt);
    yt = zeros(Nt,numObs);
    for i = 1:numObs
        yt(:,i) = i/2*(1-GJKarray(generateAllArmPolyhedra(right,qt_u,w*1,mesh_mask),obs(i),6));
    end
    yt = max(yt,[],2);
    
%     Kt = Kfun(qt_u, data_u);
    Kt = ((eye(N) - 1/N*ones(N))*(Kfun(qt_u, data_u)' - 1/N*K*ones(N,1)))';
    V = Kt*a(:,1:3);
    
    figure
    for i = 0:numObs
        plot3(V(yt==i,1), V(yt==i,2), V(yt==i,3), '.', 'color', colors(i+1,:)); hold on;
    end
    set(gca, 'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', []);
    xlabel('Component 1', 'FontSize', 10, 'Interpreter', 'latex')
    ylabel('Component 2', 'FontSize', 10, 'Interpreter', 'latex')
    zlabel('Component 3', 'FontSize', 10, 'Interpreter', 'latex')
%     for i = 0:numObs
%         plot(V(yt==i,1), V(yt==i,2), '.', 'color', colors(i+1,:)); hold on;
%     end
    grid on;
    axis vis3d;
%     title(sprintf('Kernel Type: %s', kernelType));
    set(gcf, 'color', 'w');
    if savePlot
        filename = sprintf('KPCA_baxter_%ddim_%s.gif', d, kernelType);
        if exist(filename, 'file'), delete(filename); end
        pause(0.5);
        for i = 0:3:357
            view([i 30])
            writeAnimatedGif(filename, 0.1);
        end
    end
end
%%
figure(1);
set(gcf, 'Color', 'w');
axis off;
tightfig(1);
saveas(1, '../kpcaWorkspace.png');
%%
figure(2);
set(gcf, 'Color', 'w');
saveas(2, '../kpcaCspaceFK.png');
%%
figure(3);
set(gcf, 'Color', 'w');
saveas(3, '../kpcaCspaceGauss.png');
%%
figure(4);
set(gcf, 'Color', 'w');
saveas(4, '../kpcaCspaceRQ.png');