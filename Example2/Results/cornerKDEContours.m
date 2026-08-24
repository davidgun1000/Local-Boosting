function cornerKDEContours(samplesStruct, paramNames, trueVals)
% samplesStruct: struct with fields for each method, each is [N x P] matrix of draws
%   e.g., samplesStruct.HMC_exact, samplesStruct.RVGAW, samplesStruct.HMC_Whittle
% paramNames: 1xP cell array of parameter names, e.g., {'\phi','\sigma_\eta','\sigma_\epsilon'}
% trueVals:   1xP vector of true parameter values (use [] if unknown)

methods  = fieldnames(samplesStruct);
P = numel(paramNames);
figure('Color','w'); 
tiledlayout(P,P,'Padding','compact','TileSpacing','compact');

% colors to match Figure 4 vibe: blue, red, yellow
methodColors = containers.Map(...
  methods, ...
  num2cell([0 0.447 0.741; 0.850 0.325 0.098; 0.929 0.694 0.125],2)); % adjust/order as you like

% Diagonals: 1D KDEs
for j = 1:P
    nexttile((j-1)*P + j);
    hold on
    for m = 1:numel(methods)
        S = samplesStruct.(methods{m});
        [f,xi] = ksdensity(S(:,j));
        plot(xi,f,'LineWidth',1.6,'Color',methodColors(methods{m}));
    end
    if ~isempty(trueVals)
        xline(trueVals(j),'k:','LineWidth',1.0);
    end
    xlabel(paramNames{j},'Interpreter','tex'); 
    set(gca,'YTick',[]);
    box on
end

% Off-diagonals: 2D KDE contour levels
% choose common contour levels (quantiles of density) for comparability
levels = 1; % number of contour levels

for r = 2:P
    for c = 1:r-1
        ax = nexttile((r-1)*P + c); 
        hold(ax, 'on');

        for m = 1:numel(methods)
            S = samplesStruct.(methods{m});
            x = S(:, c); 
            y = S(:, r);

            % grid for KDE
            xgrid = linspace(min(x), max(x), 120);
            ygrid = linspace(min(y), max(y), 120);
            [X, Y] = meshgrid(xgrid, ygrid);

            % evaluate bivariate KDE
            F = ksdensity([x y], [X(:) Y(:)]);
            F = reshape(F, size(X));

            % filled contours (semi-transparent)
%             [~, h] = contour(X, Y, F, levels, 'LineStyle', 'none');
%             set(h, ...
%                    'FaceColor', methodColors(methods{m}));  % adds transparency

            % add contour outlines
            contour(X, Y, F, levels, ...
                    'LineColor', methodColors(methods{m}), ...
                    'LineWidth', 0.8);
        end

        % mark true parameter (×)
        if ~isempty(trueVals)
            plot(trueVals(c), trueVals(r), 'kx', ...
                 'MarkerSize', 8, 'LineWidth', 1.6);
        end

        xlabel(paramNames{c}, 'Interpreter', 'tex'); 
        ylabel(paramNames{r}, 'Interpreter', 'tex');
        box on;
        axis([-2 2 -2 2 0 Inf]);
    end
end

% upper triangle: hide axes (to mimic classic corner plots)
for r = 1:P-1
    for c = r+1:P
        nexttile((r-1)*P + c); axis off
    end
end

% single legend
lgdAxes = nexttile(P); % reuse last tile's axes position
delete(lgdAxes);
axes('Position',[0.73 0.88 0.25 0.1],'Visible','off');
hold on
for m = 1:numel(methods)
    plot(nan,nan,'-','LineWidth',1.6,'Color',methodColors(methods{m}));
end
legend(methods,'Location','northoutside','Orientation','horizontal','Box','off');
end