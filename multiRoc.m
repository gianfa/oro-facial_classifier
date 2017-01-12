function [ fig_total_plot] = multiRoc( targets, outputs, names )
%% Effettua un plotroc multiplo basato su plot
    [ tpr, fpr, boh ] = roc( targets, outputs );
    fig_total_plot = figure();
    fig_total_plot.Color='w';
    %fig_total_plot.Name = 'Naive Bayes Roc per varianze spiegate diverse';
    %title('Naive Bayes Roc per varianze spiegate diverse')
    xlabel( 'False Positive Rate' );
    ylabel( 'True Positive Rate' );
    hold on;
    for nr = 1:length( names )
        plot( fpr{ nr }, tpr{ nr } );
    end
    plot( [0:0.1:1],[0:0.1:1],'Color','[0.8 0.8 0.8]' )
    legend( names , 'Location' ,'SouthEast');
    
end
