function spider(data, labels, ca)
% SPIDER simple spider/clock plot
%    data: 1xN array, N is the number of axes


range = squeeze(max(data));
n_axes = length(data);

rho = range/10 + zeros(1,2*n_axes+1);
rho(1:2:end-1) = data;
[x,y] = pol2cart(linspace(0, 2*pi, 2*n_axes+1), rho);
endpoints = [x(1:end-1); y(1:end-1)];

if isempty(ca)
    cf = figure();
    ca = gca(cf);
end

axes(ca);
fill(endpoints(1,:), endpoints(2,:), 'r')

if range > 0
    [label_x, label_y] = pol2cart(linspace(0, 2*pi, 2*n_axes+1), range);
    label_endpoints = [label_x(1:2:end-1); label_y(1:2:end-1)];
    for l=1:length(labels)
        if data(l)>0
            text(endpoints(1,1+2*(l-1)), endpoints(2,1+2*(l-1)), strrep(labels(l),"_"," "));
        end
    end
end
    
lim_scale = range;
if lim_scale > 0
    xlim([-lim_scale,lim_scale]);
    ylim([-lim_scale,lim_scale]);
end


end