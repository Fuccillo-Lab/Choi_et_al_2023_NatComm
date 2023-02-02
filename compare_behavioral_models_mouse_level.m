function compare_behavioral_models_mouse_level(data)

b_log = logistic_regression_behavior(data, 5, false);
FDE_log = b_log.FDE;


[b_rec, ~, ~] = recursive_logistic_regression_behavior(data, false);
FDE_rec = b_rec.FDE;

b_wl = noisy_wstlsw_behavior(data);
FDE_wl = b_wl.FDE;


gray = ones(1,3)*0.3;


for measure=["FDE", "AIC"]
    measure = char(measure);
    
    m = max([b_log.(measure); b_rec.(measure); b_wl.(measure)]);
    
    figure()
    set(gcf, 'Position', [100, 100, 560, 240])
    subplot(1,2,1)
    hold on
    plot([0,m], [0,m])
    scatter(b_rec.(measure), b_log.(measure), 'MarkerFaceColor', gray, 'MarkerEdgeColor', 'w')
    xlabel("Recursive regression " + measure)
    ylabel("Full regression " + measure)
    
    
    subplot(1,2,2)
    hold on
    plot([0,m], [0,m])
    scatter(b_wl.(measure), b_log.(measure), 'MarkerFaceColor', gray, 'MarkerEdgeColor', 'w')
    xlabel("Noisy win-stay/lose-switch " + measure)
    ylabel("Full regression " + measure)

end