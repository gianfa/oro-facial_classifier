%% Binary Classifier of oro-facial expressions
% Born as a university class project by G.Angelini e M.Raimondi.
% Class in Analysis of Biomedical Data and Signals, prof. Vittorio
% Sanguineti, University of Genoa(Italy), 2015-2016.
% 
% Edited by Gianfrancesco Angelini
% Last-update 12-01-2017
%
% 3B.Realizzazione di un sistema di riconoscimento automatico di espressioni facciali. 
% 
% 


%%
close all;



%% General Settings
% Se una di queste variabili sarà impostata nel ciclo, verranno ignorati i
% valori dati qui.

scaricareDataset = 1;           % Se 1 scarica il dataset e lo inserisce nella cartella db
url = 'db';                     % nome della cartella che contiene/conterrà il dataset
resize = 1;                     % Flag che indica se applicare un resize preliminare alle foto
resize_factor = 0.2;            % Fattore di scalatura.
metodoDiCompressione = 'pca';   % impostare 'svd','pca','none'
varSpiegata = 0.95;             % Varianza spiegata desiderata nella pca.
tic;


%% Analysis cycle settings
% Qui vanno i settaggi relativi al ciclo di analisi da effettuare.

% Impostazioni ciclo
cicla = 1;% Indica se effettuare il ciclo. 0 se non.  
variabile_ciclo = 'varSpiegata'; % Inserire il nome testuale della variabile.
valori_ciclo = [ 0.5, 0.7, 0.8, 0.9, 0.95 ]; % Il vettore con i valori che assumerà variabile_ciclo durante i cicli.
   


%% Download the dataset for training
if scaricareDataset == 1
    % ##### ONLY FOR RESEARCH PURPOSE! #####
    sourceUrl = 'http://fei.edu.br/~cet/frontalimages_manuallyaligned_part1.zip';
    [filenames] = downloadAndUnzipFromUrl( sourceUrl ); 
end


%% Ciclo di analisi

% variabili contenitore del ciclo
nb_a = [];          
nn_a = []; 
nb_p_globali = [];       
dimImg_globali = [ ]; 
k_globali = [];
img_tr_globali = {};
w_globali = [ ];
plotroc_outputSet_nb = [];
plotroc_outputSet_nn = [];
% Inizializzazione del ciclo
if cicla == 0
    valori_ciclo = [];
end

for n_global_ciclo = 1: numel( valori_ciclo );

    eval( strcat( variabile_ciclo, '= valori_ciclo( n_global_ciclo )' ) );
    %% Estrazione delle immagini

    % Le immagini vengono estratte dalla cartella specificata e catalogate in
    % sorridenti e non sorridenti, come 0 e 1, e ridimensionate.
    if resize == 1
        img_dim = [ 360, 260 ];
        img_dim = resize_factor.*img_dim;
        [ imgs_loaded, smiles, imgs_loaded_origDim ] = collectImagesAndSmilesFromUrl( url, 'jpg', [ img_dim(1), img_dim(2) ] );
        img_dim = [ imgs_loaded_origDim(1)*resize_factor, imgs_loaded_origDim(2)*resize_factor ];
        imgs = double( imgs_loaded );
    else
        [ imgs_loaded, smiles, imgs_loaded_origDim ] = collectImagesAndSmilesFromUrl( url, 'jpg' );
        img_dim = [ imgs_loaded_origDim(1)*resize_factor, imgs_loaded_origDim(2)*resize_factor ];
        imgs = double( imgs_loaded );
    end
    
    %% LDA e features

    if n_global_ciclo == 1
        nImg_lda = 120;
        lda_set_labels = smiles( 1: nImg_lda );
        lda_set_imgs = imgs';                   % [immagini x features]

        lda_set_imgs = lda_set_imgs( :, 1: nImg_lda );
        %ripartisco le immagini secondo la classe e calcolo i vettori medi per ogni classe
        lda_set_labels_pos_index = find( lda_set_labels );
        lda_set_imgs_pos = lda_set_imgs( :, lda_set_labels_pos_index );
        lda_set_imgs_pos_media = mean( lda_set_imgs_pos' );
        lda_set_labels_neg_index = find( not(lda_set_labels) );
        lda_set_imgs_neg = lda_set_imgs( :, lda_set_labels_neg_index );
        lda_set_imgs_neg_media = mean( lda_set_imgs_neg' );

        mu_S = lda_set_imgs_neg_media; % media immagini no
        mu_H = lda_set_imgs_pos_media; % media immagini yes

        sigma_S = diag( std( lda_set_imgs_pos' ).^2); % varianze 
        sigma_H = diag( std( lda_set_imgs_neg' ).^2); % varianze 

        Sw_S = ( length( lda_set_labels_pos_index ) -1 )*sigma_S; 
        Sw_H = ( length( lda_set_labels_pos_index ) -1 )*sigma_H; % quantità di sorridenti
        Sw = Sw_S+Sw_H;

        w = inv(Sw)*( mu_H'-mu_S' );
        modw = sqrt( sum(w.^2) );        % norma di w
        w = w./modw;
        w_globali = [ w_globali; w' ];   % conservo il valore attuale in una variabile esterna al ciclo
        


        %% Classificatore 2.0 : LDA

        lda_input = double( imgs_loaded )';
        lda_train_nImg = 120;
        lda_train_input = lda_input( 1:lda_train_nImg, : ); %[osservazioni x features]
        lda_train_targetSet = smiles( 1:lda_train_nImg ); %[osservazioni]

        lda_test_imgs = lda_input( :,lda_train_nImg+1:end );
        lda_test_labels=smiles(lda_train_nImg+1:end);
        prior_S = length( lda_set_imgs_neg(1,:) )./length( lda_set_labels ); % quantità yes su tot , training
        prior_H = length( lda_set_imgs_pos(1,:) )./length( lda_set_labels );

        Z_tr = lda_set_imgs'* w;

        zmu_S = mean( Z_tr( lda_set_labels_neg_index ));
        zmu_H = mean( Z_tr( lda_set_labels_pos_index ));
        zsigma_S = cov(Z_tr( lda_set_labels_neg_index ));
        zsigma_H = cov(Z_tr( lda_set_labels_pos_index ));

        % test del classificatore
        Z_te = lda_test_imgs'*w;
        p_z_isS = mvnpdf( Z_te, zmu_S, zsigma_S ); %p(z|nonS)
        p_z_isH = mvnpdf( Z_te, zmu_H, zsigma_H );

        p_isS_z = p_z_isS.*prior_S./( p_z_isS.*prior_S+p_z_isH.*prior_H );
        p_isH_z = p_z_isH.*prior_H./( p_z_isS.*prior_S+p_z_isH.*prior_H );

        myLDAchoice = p_isH_z > p_isS_z;
        myLDAchoice = double(myLDAchoice); 

        LDA_res = [lda_test_labels', myLDAchoice]; 
    end




%% Compressione dei dati


if strcmp( lower( metodoDiCompressione ),'svd') == 1
    % Compressione 1 : SVD su ogni immagine
    % comprimo le immagini del set, conservando in un array la matrice di
    % rotazione(autovettori) della compressione
%    k = 0;
    [ imgs, imgs_autovettori, imgs_out ] = comprimiSet( imgs', k, [ img_dim(1),img_dim(2) ] );
    imgs = imgs';
    img_tr_globali = [ img_tr_globali, imgs_out( 1 ) ];
    
elseif strcmp( lower( metodoDiCompressione ),'pca' ) == 1
    % Compressione 2 : PCA sul set di immagini
    
    [ coeff, score, latent ] = pca( imgs );
    % calcolo delle k features corrispondenti alla varianza spiegata data
    expVar = cumsum(latent)./sum(latent);
    k = max ( find( expVar <= varSpiegata ) );
    
    imgs = score( : , 1: k );
    
elseif strcmp( lower( metodoDiCompressione ),'none' ) == 1
    % Nessuna compressione
else
    error('Errore linea 6: Metodo di compressione non corretto.');
end



%% Classificatori
% Vengono inizializzati due classificatori tramite utilizzo di PCA: una rete
% neurale multistrato ed un classificatore bayesiano naive.

%% Classificatore 1 : Rete neurale
% Net è l'oggetto rete, y l'output ottenuto dalla rete
%input2=projections_X(:,1:3)'*input;

nTrainImg = 120;                                    % n di immagini da usare per addestrare la rete
nn_train_inputSet = imgs( 1: nTrainImg, : )';       % immagini con le quali addestrare la rete neurale
nn_train_targetSet = smiles( 1: nTrainImg );        % vettore di labels, definite da 0 o 1, riguardanti le immagini in nn_train_inputSet
nn_neuroni = 1;                                     % neuroni della rete neurale

% costruzione dell'oggetto rete  
net = patternnet( nn_neuroni ); 
% addestramento della rete
net = train( net, nn_train_inputSet, nn_train_targetSet );
v = view(net);
v.hide;
nntraintool('close');
nn_train_risultati = net( nn_train_inputSet );
nn_perf = perform( net, nn_train_targetSet, nn_train_risultati );
classes = vec2ind( nn_train_risultati );
%save('smileRecognizer');

% calcoliamo l'area della curva ROC della rete neurale
[ aSpecifity_net, sensitivity_net, T_net, auc_net] = perfcurve( nn_train_targetSet, nn_train_risultati, 1);
% Elaborazione performance
%nn_soglia = 0.1;
nn_train_risultati_round = nn_train_risultati;
nn_train_risultati_round( nn_train_risultati_round <= 0.5 ) = 0;
nn_train_risultati_round( nn_train_risultati_round > 0.5 ) = 1;
nn_cp = classperf( nn_train_targetSet, nn_train_risultati_round );
nn_train_perf_dati = [ nn_cp.errorrate, nn_cp.sensitivity, nn_cp.specificity  ];
%nn_totPerform = nn_train_perf_dati;
%[ nn_errParted, nn_negFeat, nn_posFeat, nn_posFeat_limits ] = elaboraPerf ( nn_train_risultati_round, nn_soglia );



%% Classificatore 2 : Bayesiano Naive

nb_ni = 120;                                % n di immagini input per l'addestramento del classifictore
nb_input = imgs( 1:nb_ni, : );              %
nb_train_targetSet = smiles( 1:nb_ni );

% Inizializiamo il classificatore
nb_class = fitNaiveBayes( nb_input, nb_train_targetSet );

% Analizziamo le performance del naive Bayes
nb_testSet = imgs( ( nb_ni+1 ):end, : ); %[ oss x feat ]
[ nb_p, nb_results] = nb_class.posterior( nb_testSet );
nb_results = nb_results';

nb_test_targetSet = smiles( : ,(nb_ni+1):end );

nb_plotroc_outputSet = nb_p(:,2)';
%[trp, fap, th] = roc( targetSet_nB , risultati_nB );

%fig_nb_roc = figure();
%fig_nb_roc.Name = 'Roc Naive Bayes';
%plotroc( nb_test_targetSet, nb_p(:,2)' );
%fig_nb_roc.Position = [ 10, 10,  250, 250];
%title(' Roc Naive Bayes ');
[ aSpecifity_nB_cum, sensitivity_nB_cum, T_nB, auc_nB] = perfcurve( nb_test_targetSet, nb_results, 1);

%%
    
    nn_accuracy = 1-nn_cp.errorrate;
    nn_a = [nn_a, nn_accuracy];
    nb_er = nb_test_targetSet+nb_results ;
    nb_er( nb_er ~= 1 ) = 0;
    nb_er_v = sum(nb_er)/numel(nb_er);
    nb_accuracy = 1-nb_er_v;
    nb_a = [nb_a, nb_accuracy];
    plotroc_outputSet_nb = [ nb_plotroc_outputSet ; plotroc_outputSet_nb ]; 
    plotroc_outputSet_nn = [nn_train_risultati ; plotroc_outputSet_nn];
    % salvo alcune variabili al di fuori del ciclo globale
    k_globali = [ k_globali, k];
    nb_p_globali = [ nb_p_globali; nb_p ];
    dimImg_globali = [ dimImg_globali; img_dim ];
    disp( sprintf('Concluso il ciclo %d/%d', n_global_ciclo, numel( valori_ciclo )) );
    
    
   
end %Fine del ciclo di analisi

%% Visualizzazione dei risultati del ciclo

% Plotroc della Naive Bayes con varianze spiegate variabili
plotroc_targetSet_nb = ones( 1, numel(valori_ciclo) )' * nb_test_targetSet;
%names_roc = { '0.95' ; '0.90'; '0.80' ; '0.70'; '0.50' };
names_roc = {};
for n_pn = 1:numel( valori_ciclo )
    names_roc = [ names_roc, num2str( valori_ciclo( n_pn ) ) ];
end
fig_nb_multi=multiRoc( plotroc_targetSet_nb, plotroc_outputSet_nb , names_roc);
fig_nb_multi.Name = 'Roc : Naive Bayes per exVariance';
title( strcat('ROC Naive Bayes al variare di ', variabile_ciclo) );
fig_nb_multi_legend = legend(names_roc);
fig_nb_multi_legend = fig_nb_multi.Children(1);
fig_nb_multi.Position = [800, 50, 400,300];

% Plotroc della Neural Network con varianze spiegate variabili
plotroc_targetSet_nn=ones(1, numel(valori_ciclo))' * nn_train_targetSet;
fig_nn_multi=multiRoc( plotroc_targetSet_nn , plotroc_outputSet_nn, names_roc);
fig_nn_multi.Name = 'Roc : Neural Network per exVariance';
title( strcat('ROC Neural Network al variare di ', variabile_ciclo) );
fig_nn_multi_legend = legend(names_roc);
fig_nn_multi_legend = fig_nn_multi.Children(1)
fig_nn_multi.Position = [375, 35, 350, 350];

% Accuratezza dei classificatori
x_fig_acc = [];
y_fig_acc = 0.1:0.1:0.7;

for n_2 = 1:size( nn_a,2 )
    x_fig_acc = [ x_fig_acc; [ nn_a(n_2), nb_a(n_2) ]  ];
end

fig_acc = figure();
fig_acc.Name = 'Accuratezza dei classificatori';
fig_acc.Color = 'w';
labels = { 'NNetwork' , 'Naive Bayes' };
barre = bar( x_fig_acc,'DisplayName','x' );
legend( barre, labels, 'Location','northwest' );
legend show;
title('Accuratezza dei classificatori', 'fontSize', 16);
set(gca,'XTickLabel',{ valori_ciclo });
xlabel( variabile_ciclo );
ylabel( 'Success Rate');
fig_acc.Position = [ 800, 380, 400, 300 ];

% Storia pesi LDA
showHistory_LDA = 0;

if showHistory_LDA == 1
    fig_lda_weights = figure();
    fig_lda_weights.Name = ('LDA Features');
    title('LDA Features', 'fontsize', 18);
    l_3 = size( w_globali,1 );
    for n3 = 1 : l_3
        subplot( 1, l_3 , n3 );
        bar( w_globali( n3, :) );
        title( num2str( valori_ciclo(n3) ) );
        hold on;
    end
end

% LDA: 1_Features weights
fig_lda_features = figure();
fig_lda_features.Name = ('LDA: Features');
fig_lda_features.Position = [ 20, 400, 440,280 ];
bar( w );
title('LDA Features');
xlabel('features');
ylabel('features class weights');

% LDA: 2_Features img
%   Vengono visualizzate le features più rilevanti, in base alla
%   magnitudine del parametro (o peso) ottenuto dall'LDA
fig_lda_vis = figure();
fig_lda_vis.Name = 'LDA: Visualizzazione features';
% Prelievo di un'immagine campione
lda_im1_orig = double( imgs_loaded(1,:) );
% Normalizzazione dei valori dei parametri dell'LDA
w1 = w.*( ones(  numel(w), 1 ))./sum(w);
% Pesatura delle feature con i parametri normalizzati
lda_im1 = w1'.* lda_im1_orig;
% Traslazione del minimo, per renderle visibili all'occhio umano
minimi = ones( 1, img_dim(1)*img_dim(2) ) .* abs( min(lda_im1) );
lda_im1 = lda_im1 + minimi;
lda_im1 = reshape( lda_im1, img_dim(1),img_dim(2) );
% Visualizzazione
subplot( 1,2,1 );
subimage( uint8( reshape (lda_im1_orig, img_dim(1),img_dim(2) ) ) );
title('Originale');
axis off;
subplot( 1,2,2 );
subimage( double( lda_im1 ) );
title('LDA best features (bianco)');
axis off;

% LDA: 3_Roc
fig_lda_roc = figure();
plotroc( lda_test_labels, p_isH_z' );
title('ROC LDA');
fig_lda_roc.Position = [ 310, 50,  250, 250 ];

% LDA : 4_Confusion Matrix 
confusion_matrix_LDA_te = confusionmat(lda_test_labels,myLDAchoice);
fig_lda_confusion = figure();
fig_lda_confusion.Name = 'LDA Confusion Matrix';
fig_lda_confusion.Position = [ 10, 400, 240,280 ];
fig_conf = plotconfusion( lda_test_labels , myLDAchoice' );
title('LDA Confusion Matrix');
fig_conf.Position = [20, 50, 250, 250];



% SVD: Features considerate
if strcmp( lower( metodoDiCompressione ),'svd' ) == 1
    fig_face = figure();
    fig_face.Color =  'w';
    fig_face.Name = 'SVD features';
    title('Features SVD','fontsize',18);
    axis off;
    svd_totImg = numel( k_globali );
    svd_cols = 4;
    svd_sub_rows = ceil( svd_totImg/svd_cols );
    
    for n_g1 = 1 : svd_totImg
        if n_g1 == 1
            img_cur = uint8( reshape ( imgs_loaded( 1,: ), img_dim(1), img_dim(2) ) );
        else
            % ricostruzione dell'osservazione nel sistema di rif originale
            img_cur = uint8( cell2mat( img_tr_globali( n_g1 ) ) );
        end
        % inserimento nella window
        subplot( svd_sub_rows, svd_cols, n_g1 );
        subimage ( img_cur );
        if n_g1 == 1
            title( 'originale' );
        else
            title( valori_ciclo( n_g1 ) );
        end    
        axis off;
    end
    fig_face.Position = [600,300,500,400];
end

% PCA: Features considerate
if strcmp( lower( metodoDiCompressione ),'pca' ) == 1
    fig_face = figure();
    fig_face.Color =  'w';
    fig_face.Name = 'PCA features';
    title('Features PCA','fontsize',18);
    axis off;
    svd_totImg = numel( k_globali );
    for n_g1 = 1 : svd_totImg
        if n_g1 == 1
            img_cur = uint8( imgs_loaded( 1,: ) );
        else
            % ricostruzione dell'osservazione nel sistema di rif originale
            k_cur = k_globali( n_g1 );
            %coeff_cur = coeff( :, 1:k_cur );
            img_cur = score( 1, 1:k_cur ) * coeff( :, 1:k_cur )' ;
        end
        % ricostruzione dell'immagine
        img_cur = reshape( img_cur, dimImg_globali( n_g1, 1 ), dimImg_globali( n_g1, 2 ) );
        % inserimento nella window
        subplot( 1, svd_totImg, n_g1 );
        subimage(  img_cur  );
        if n_g1 == 1
            title( 'originale' );
        else
            title( valori_ciclo( n_g1 ) );
        end    
        axis off;
    end
    fig_face.Position = [600,300,500,200];
end


% controllo che tutte le figure abbiano almeno lo sfondo bianco, per
% renderle più gradevoli alla vista
tutteLeFigure = findall(0,'type','figure');
for n = 1:numel(tutteLeFigure)
fig_cur = tutteLeFigure(n);
fig_cur.Color = 'w';
end

txt_auc = sprintf('Fine esecuzione');
disp( strcat( txt_auc ) );




execTime = toc;
disp( execTime );




    