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

% rng(7);
rng(0);
time_delay = 0.05;
d = 2;
L = repmat(Link('d', 0, 'a', 1, 'alpha', 0), d, 1);

R = SerialLink(L);
axLim = (sum(R.a) + 1)*[-1 1 -1 1 -1 1];
R.base = trotx(pi/2)*R.base;

R.plotopt = {'view', 'x', 'noshading', 'noname', 'ortho', ...
    'workspace', axLim, 'tile1color', [1,1,1], 'delay', time_delay, 'trail', '', ...
    'noshadow', 'nobase', 'nowrist', ...
    'linkcolor', 'b', 'toolcolor', 'b', ...
    'jointcolor', 0.2*[1 1 1], 'jointdiam', 2, 'jointscale', 1, 'scale', 0.5};
axis(axLim); axis square; hold on;
opt = R.plot(zeros(1, d));
w = 2*opt.scale; % get width of robot arm based on its size in the figure
%%
colors = [0 0.6 0; 0.5 1 0.9; 0.25 0.25 1; 1 0.6 0.8];
obs{1} = 3*[-0.1 -0.1; 0.1 -0.1; 0.1 0.1; -0.1 0.1];

offset = [0.8 1.1];
numObs = 1;
obs{1} = obs{1} + offset;

for i = 1:numObs
    [x,y] = pol2cart(2*pi*rand, 0.8+0.4*rand);
    obs{i} = (obs{1} - offset) + [x,y];
end

obs_color = [1 0.3 0.3];
robot_color = [0 0.5 1];

colors = cbrewer('qual', 'Set1', numObs + 1);
        
for i = 1:numObs
    fill3(obs{i}(:,1), -ones(size(obs{i}(:,1))), obs{i}(:,2), colors(i+1,:), 'linewidth', 2);
end
%%
N = 1000;
q = pi*(2*rand(N,d) - 1);

y = ones(N,1);
colCheck = @(x) ~gjk2Darray(generateArmPolygons(R, x, w), obs);
for i = 1:N
	y(i) = colCheck(q(i,:));
end
%%
cm = interp1([0 0.5 1]', [0.4039 0.6627 0.8118; 0.9686 0.9686 0.9686;0.9373 0.5412 0.3843;], linspace(0,1,256)');
% cm = cbrewer('div', 'RdBu', 256);

method = 'RBF';
% method = 'FK';

if strcmp(method, 'RBF')
    Kfun = @(X,Y) exp(-1 * pdist2(X, Y).^2);
    lambda = 0.1;
else
    Kfun = @(X,Y) robotKernel(R,X,Y,1);
    lambda = 0.1;
end

K = Kfun(q, q);

% a = trainFastron(q, 2*y-1, 50000, N, 1, 1, zeros(N,1), zeros(N,1), K);
% q = q(a~=0,:);
% y = y(a~=0);
% K = K(a~=0,a~=0);
[a, S, idx] = ivmTrain2(q, y, K, lambda);
%%
[X,Y] = meshgrid(linspace(-pi,pi,100), linspace(-pi,pi,100));
p = zeros(size(X));

F = Kfun([X(:) Y(:)], S)*a(1:end-1) + a(end);

% F = Kfun([X(:) Y(:)], S)*a;
p(:) = 1./(1 + exp(-F));
% p(:) = max(p, 1-p);

figure;
imagesc([-pi pi], [-pi pi], imresize(p, 4), [0 1]); hold on;
set(gca, 'ydir', 'normal');
% imagesc([-pi pi], [-pi pi], imresize(p, 4)>0.75 | imresize(p, 4)<0.25, [0 1]); hold on;

% plot(q(y~=0, 1), q(y~=0, 2), 'r.', 'markersize', 2); hold on;
% plot(q(y==0, 1), q(y==0, 2), 'b.', 'markersize', 2);

plot(S(:,1), S(:,2), 'k.');
plot(S(:,1), S(:,2), 'ko', 'MarkerSize', 5);
axis square;

colormap(cm);
% viscircles([0 0], 0.5, 'color', 'g', 'linewidth', 1, 'EnhanceVisibility', false);

[cont, cont_h] = contour(X, Y, p);
cont_h.LevelList = [0.1 0.25 0.5 0.75 0.9];
cont_h.LevelList = [0.1 0.5 0.9];
% cont_h.LevelList = [0.1 0.5 0.9 0.95 0.99];
cont_h.LineColor = 'k';
clabel(cont, cont_h, 'Color', 'k', 'FontSize', 10, 'interpreter', 'latex');

% title('P(Y = 1|X = x)');

% imgGt = zeros(size(X));
% for i = 1:numel(imgGt)
%     imgGt(i) = colCheck([X(i) Y(i)]);
% end
% [~,cont_h] = contour(X, Y, imgGt);
% cont_h.LevelList = [0.5];
% cont_h.LineColor = 'r';

set(gcf, 'Position', [681 481 477 468], 'color', 'w');

set(gca, 'XTick', linspace(-pi,pi,3), 'YTick', linspace(-pi,pi,3));
set(gca, 'FontSize', 15)
set(gca, 'XTickLabels', {'$$-\pi$$', '0', '$$\pi$$'}, 'YTickLabels', {'$$-\pi$$', '0', '$$\pi$$'}, 'ticklabelinterpreter', 'latex');
tightfig(2);

saveas(2, sprintf('../ivm%s.png', method))
%%
% [gx, gy] = imgradientxy(p);
% quiver(X, Y, gx, gy, 10)