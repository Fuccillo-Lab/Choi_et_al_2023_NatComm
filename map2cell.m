function [pred_names, data] = map2cell(map)
% MAP2CELL extract data from a Map into a cell array
%
%   Typically, the argument |map| will contain something like a list of
%   kernels, indexed by the predictor they are associated with. This
%   function takes this map as input, and outputs two cell arrays. The
%   first (|pred_names|) is a cell array containing all predictor names,
%   and the second (|data|) is a cell array containing the contents of the
%   map (so for instance a cell array where each cell contains the values
%   that make up a kernel).
%
%   Example
%
%   % Extract population kernels for aDMS. kernels, cis and lags will be
%   % contained in maps.
%   [pop_kernels, pop_kernels_ci, pop_kernels_lags] = e.populations('aDMS').get_population_kernels();
%   % extract the kernels and the lags in cell arrays
%   [predictor_names, kernels_array] = map2cell(pop_kernels);
%   [~, lags_array] = map2cell(pop_kernels_lags);
%   % plot the 4th kernel (whatever that is) and display the name of its
%   % associated predictor
%   plot(lags_array{4}, kernels_array{4});
%   title("Population kernel for predictor: %s", predictor_names{4});

pred_names = keys(map)';
data = {};
for p=1:length(pred_names)
    name = pred_names{p};
    data = [data; map(name)];
end

end