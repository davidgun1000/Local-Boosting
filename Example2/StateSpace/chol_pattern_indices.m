function idx = chol_pattern_indices(n, d, mG)
% chol_pattern_indices  Build indices for sparse Cholesky L in Eq. (12)
%   n  = number of time points (states), e.g. 1000
%   d  = state dimension, e.g. 2
%   mG = number of global params (can be 0 if none)
%
% Returns struct idx with:
%   .I, .J         : row/col indices for nonzeros of L
%   .S             : logical pattern (sparse) of L
%   .N             : total dimension of L
%   .par           : parameter vector layout matching paper notation:
%                    par.Lii{i}      -> vech(L_i) (size d(d+1)/2)
%                    par.Ltilde{i}   -> vec(Ltilde_i) for i=1..n-1 (size d*d)
%                    par.LGi{i}      -> vec(L_Gi) (size mG*d)
%                    par.LG          -> vech(L_G) (size mG(mG+1)/2)

Nstate = n*d;
N      = Nstate + mG;

I = []; J = [];

% Helper: append lower-triangular (including diag) block positions
    function [Ii,Ji] = tri_lower_block(rows, cols, dim)
        [cc, rr] = meshgrid(1:dim,1:dim);
        mask = rr >= cc;                 % lower triangle incl diag
        rr = rows(rr(mask));
        cc = cols(cc(mask));
        Ii = rr(:); Ji = cc(:);
    end

% Helper: append full block positions
    function [Ii,Ji] = full_block(rows, cols)
        [cc, rr] = meshgrid(cols, rows);
        Ii = rr(:); Ji = cc(:);
    end

% Precompute row/col ranges for each state block
rows_blk = @(i) ((i-1)*d + (1:d));
cols_blk = rows_blk; % square

% Parameter vector bookkeeping (paper’s ordering)
par = struct(); par.Lii = cell(n,1); par.Ltilde = cell(max(n-1,0),1);
par.LGi = cell(n,1); par.LG = [];
p = 0;

% 1) State diagonal blocks: L_i (lower-triangular dxd), i=1..n
for i = 1:n
    r = rows_blk(i); c = cols_blk(i);
    [Ii,Ji] = tri_lower_block(r,c,d);
    I = [I; Ii]; J = [J; Ji];

    % parameter layout: vech(L_i)
    par.Lii{i} = p + (1:(d*(d+1)/2));  p = p + numel(par.Lii{i});
end

% 2) State sub-diagonal links: Ltilde_i at (i+1, i), full dxd, i=1..n-1
for i = 1:n-1
    r = rows_blk(i+1); c = cols_blk(i);
    [Ii,Ji] = full_block(r,c);
    I = [I; Ii]; J = [J; Ji];

    % parameter layout: vec(Ltilde_i)
    par.Ltilde{i} = p + (1:(d*d));  p = p + numel(par.Ltilde{i});
end

% 3) Global–state blocks: L_Gi (mG x d), full, at (G, i), i=1..n
if mG > 0
    grow = Nstate + (1:mG);
    for i = 1:n
        c = cols_blk(i);
        [Ii,Ji] = full_block(grow, c);
        I = [I; Ii]; J = [J; Ji];

        % parameter layout: vec(L_Gi)
        par.LGi{i} = p + (1:(mG*d));  p = p + numel(par.LGi{i});
    end

    % 4) Global block L_G (mG x mG), lower-triangular
    [Ii,Ji] = tri_lower_block(grow, grow, mG);
    I = [I; Ii]; J = [J; Ji];

    % parameter layout: vech(L_G)
    par.LG = p + (1:(mG*(mG+1)/2));  p = p + numel(par.LG);
else
    % no globals
    par.LGi(:) = {[]};
    par.LG = [];
end

% Build logical sparse pattern (values not assigned here)
S = sparse(I, J, true, N, N);  %#ok<SPRIX>

% Output
idx.I = I; idx.J = J; idx.S = S; idx.N = N; idx.par = par;
end