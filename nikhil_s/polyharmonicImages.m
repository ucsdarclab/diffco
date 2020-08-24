if ispc
    addpath(genpath('C:/Users/nkhld/Dropbox/ARClab/TwoLink/GaussianBeliefSpaceTwoLink'));
    addpath('C:/Users/nkhld/Dropbox/MATLAB/');
    addpath('C:/Users/nkhld/Dropbox/MATLAB/cbrewer/');
    addpath('C:/Users/nkhld/Dropbox/ARClab/TwoLink/GaussianBeliefSpaceTwoLink/DRRT');
end

rng(7);
g = 10;
load example2DOFdata;
switch 1
    case 1
        tunnel_width = 01; tunnel_length = 1;
        data(data(:,2)<tunnel_length/2 & data(:,2)>-tunnel_length/2,end) = -1;
        data(data(:,1)<tunnel_width/2 & data(:,1)>-tunnel_width/2,end) = 1;
        data(data(:,1)<-tunnel_width/2 & data(:,1)>-3*tunnel_width/2,end) = -1;
        data(data(:,2)>tunnel_length/2 | data(:,2)<-3*tunnel_length/2,end) = 1;
        viewpt = [-44.3657 -15.8854 463.0127];
        
        r = randsample(1:size(data,1),4000);
        data = data(r,:);
    case 2
        data = dlmread('CspaceSampleData.txt');
        data(data(:,1)>pi,1) = data(data(:,1)>pi,1) - 2*pi;
        data(data(:,2)>pi,2) = data(data(:,2)>pi,2) - 2*pi;
        viewpt = [9.5814 52.7238 450.9593];
        
        r = randsample(1:size(data,1),4000);
        data = data(r,:);
    case 3
        data = dlmread('Cspace1.txt');
        data(data(:,1)>pi,1) = data(data(:,1)>pi,1) - 2*pi;
        data(data(:,2)>pi,2) = data(data(:,2)>pi,2) - 2*pi;
        
        viewpt = [-44.3657 -15.8854 463.0127];
end

a = trainPerceptronMATLAB4(data, 1000, 10);
a(a==1) = 0; % remove outliers
%%
c = data(a~=0,1:end-1)';
vals = a(a~=0)';
% xy = data(:,1:end-1)';
% vals = a';

maxTrials = 100;

close all;
% figure(1); set(gcf, 'Position', [100 246 1249 592]);
% plot(data(a<0,1), data(a<0,2), 'r.');
% plot(data(a>0,1), data(a>0,2), 'b.');
gaus = 0;
if gaus
    filename = 'gausSurface.gif';
else
    filename = 'polyharmonicSurface.gif';
end
% if exist(filename, 'file') == 2, delete(filename); end
delayTime = 0.05;
%%
switch 4
    case 4
        figure(1); set(gcf, 'Position', [100 255 611 530], 'color', 'w');
        if gaus
            img = (make2DCspace(data(a~=0,:), a(a~=0), [256 256],g));
            img_h = imagesc([-pi pi], [-pi pi], sign(img)); set(gca, 'ydir', 'normal'); hold on;
            axis square;
        else
            k = 1;
            w = fitPolyharmonicSpline(c', vals', k);
            
            [X,Y] = meshgrid(linspace(-pi,pi,200), linspace(-pi,pi,200));
            
            r = pdist2([X(:) Y(:)], c');
%             phi = r.^2.*log(r);
%             phi = r.^1;
            if mod(k,2) % odd
                phi = r.^k;
            else % even
                phi = r.^k.*log(r);
            end
            phi(isnan(phi)) = 0;
            
            Z = zeros(size(X));
            Z(:) = phi*w(1:size(c,2)) + [X(:) Y(:)]*w(size(c,2)+1:end-1) + w(end);
            %             h = surf(X,Y,Z); hold on;
            img = Z;
            img_h = imagesc([-pi pi], [-pi pi], inf*sign(Z));
            hold on; axis square;
        end
        
        colormap([1 0.7 0.7; 0.7 0.7 1]);
        
        xlabel('Joint 1 Angle (degrees)'); ylabel('Joint 2 Angle (degrees)');
        set(gca, 'XTick', linspace(-pi,pi,7), 'YTick', linspace(-pi,pi,7), ...
            'XTickLabel', linspace(-180, 180, 7), 'YTickLabel', linspace(-180, 180, 7), 'ydir', 'normal', ...
            'ZTick', []);
        
        xlim([-pi pi]); ylim([-pi pi]); zlim([min(img(:)) max(img(:))]);
        views = [linspace(0,-51,40)', linspace(90,28,40)'];
        
%         writeAnimatedGif(filename, 3);
%         for i = 1:length(views)
%             view(views(i,:))
%             %             drawnow;
%             writeAnimatedGif(filename, delayTime);
%         end
%         error
%         figure(2); set(gcf, 'Position', [800 255 611 530]);
        %         subplot(1,2,2);
        
%         writeAnimatedGif(filename, 0.5);
        [X1,Y1] = meshgrid(linspace(-pi,pi,size(img,2)),linspace(-pi,pi,size(img,1)));
        caxis([-eps eps]);
        
        figure(2); set(gcf, 'Position', [100 255 611 530], 'color', 'w');
        
        [FX, FY] = gradient(img);
        
        [X,Y] = meshgrid(linspace(-pi,pi,size(FX,2)/16),linspace(-pi,pi,size(FX,1)/16));
        FX = interp2(linspace(-pi,pi,size(FX,2))',linspace(-pi,pi,size(FX,1)), FX, X, Y);
        FY = interp2(linspace(-pi,pi,size(FY,2))',linspace(-pi,pi,size(FY,1)), FY, X, Y);
        %     diffNorm = sqrt(FX.^2+FY.^2);
        
        img_h = imagesc([-pi pi], [-pi pi], (img)); set(gca, 'ydir', 'normal');
        cm = flipud(interp1([0 0.5 1]', [0.4039 0.6627 0.8118; 0.9686 0.9686 0.9686;0.9373 0.5412 0.3843;], linspace(0,1,256)'));
        colormap(cm);
        hold on;
        
        [~, cont_h] = contour(X1, Y1, img);
        cont_h.LevelList = 0;
        cont_h.LineColor = 'k';
        
%         xlabel('Joint 1 Angle (degrees)'); ylabel('Joint 2 Angle (degrees)');
        set(gca, 'XTick', linspace(-pi,pi,3), 'YTick', linspace(-pi,pi,3));%, ...
%             'XTickLabel', {}, 'YTickLabel', linspace(-180, 180, 7),'TickLabelInterpreter', 'latex');
        set(gca, 'XTickLabels', {'$$-\pi$$', '0', '$$\pi$$'}, 'YTickLabels', {'$$-\pi$$', '0', '$$\pi$$'}, 'ticklabelinterpreter', 'latex', 'FontSize', 15);
        axis square;
        %     quiver(X,Y,FX./diffNorm,FY./diffNorm,1,'k');
        %     startX = 2*pi*rand(500,1) - pi;
        %     startY = 2*pi*rand(500,1) - pi;
        startX = data(data(:,end)<0,1);
        startY = data(data(:,end)<0,2);
        skip = 3;
        h = streamline(X,Y,FX,FY,startX(1:skip:end), startY(1:skip:end), [0.1 1e5]);
        set(h, 'Color', 0.6*[0 0 1], 'linewidth', 1);
        plot(startX(1:skip:end), startY(1:skip:end), 'r.', 'MarkerSize', 8);
   
%         tightfig(2);
%         if gaus
%             saveas(2, '../gausStreamline.png');
%         else
%             saveas(2, '../polyharmonicStreamline.png');
%         end
end
error
%%
figure(3); set(gcf, 'Position', [100 255 611 530], 'color', 'w');

% [FX, FY] = gradient(img);
% 
% [X,Y] = meshgrid(linspace(-pi,pi,size(FX,2)/16),linspace(-pi,pi,size(FX,1)/16));
% FX = interp2(linspace(-pi,pi,size(FX,2))',linspace(-pi,pi,size(FX,1)), FX, X, Y);
% FY = interp2(linspace(-pi,pi,size(FY,2))',linspace(-pi,pi,size(FY,1)), FY, X, Y);
%     diffNorm = sqrt(FX.^2+FY.^2);

img_h = imagesc([-pi pi], [-pi pi], sign(img)); set(gca, 'ydir', 'normal');
hold on;
colormap([1 0.7 0.7; 0.7 0.7 1]);

axis square;
K = @(x,y) exp(-g*sum((x-repmat(y,size(x,1),1)).^2,2));
syms x y;
hyperplane = ezplot(sum(a(a~=0).*K(data(a~=0,1:end-1),[x y])),[-pi pi -pi pi]);
title([]);
set(hyperplane, 'linecolor', 'k', 'linewidth', 1);
l(1) = plot(NaN,NaN,'sk','MarkerFaceColor',[1 0.7 0.7],'MarkerSize', 12);
l(2) = plot(NaN,NaN,'sk','MarkerFaceColor',[0.7 0.7 1],'MarkerSize', 12);
legend(l,{'C_{obs}','C_{free}'},'Location','Northwest','FontSize',15);
xlabel('Joint 1 Angle (degrees)'); ylabel('Joint 2 Angle (degrees)');
set(gca, 'XTick', linspace(-pi,pi,7), 'YTick', linspace(-pi,pi,7), ...
    'XTickLabel', linspace(-180, 180, 7), 'YTickLabel', linspace(-180, 180, 7));
saveas(3, 'cspaceExample.png')
%%
figure(4); set(gcf, 'Position', [100 255 611 530], 'color', 'w');

% [FX, FY] = gradient(img);
% 
% [X,Y] = meshgrid(linspace(-pi,pi,size(FX,2)/16),linspace(-pi,pi,size(FX,1)/16));
% FX = interp2(linspace(-pi,pi,size(FX,2))',linspace(-pi,pi,size(FX,1)), FX, X, Y);
% FY = interp2(linspace(-pi,pi,size(FY,2))',linspace(-pi,pi,size(FY,1)), FY, X, Y);
%     diffNorm = sqrt(FX.^2+FY.^2);

img_h = imagesc([-pi pi], [-pi pi], sign(img)); set(gca, 'ydir', 'normal');
hold on;
colormap([1 0.7 0.7; 0.7 0.7 1]);

axis square;
K = @(x,y) exp(-g*sum((x-repmat(y,size(x,1),1)).^2,2));
syms x y;
hyperplane = ezplot(sum(a(a~=0).*K(data(a~=0,1:end-1),[x y])),[-pi pi -pi pi]);
title([]);
set(hyperplane, 'linecolor', 'k', 'linewidth', 1);
l(1) = plot(NaN,NaN,'sk','MarkerFaceColor',[1 0.7 0.7],'MarkerSize', 12);
l(2) = plot(NaN,NaN,'sk','MarkerFaceColor',[0.7 0.7 1],'MarkerSize', 12);
legend(l,{'C_{obs}','C_{free}'},'Location','Northwest','FontSize',15);
xlabel('Joint 1 Angle (degrees)'); ylabel('Joint 2 Angle (degrees)');
set(gca, 'XTick', linspace(-pi,pi,7), 'YTick', linspace(-pi,pi,7), ...
    'XTickLabel', linspace(-180, 180, 7), 'YTickLabel', linspace(-180, 180, 7));
% saveas(3, 'cspaceExample.png')
