%% Recupera da un path tutti i files che contiene, di estensione ext.
% Restituisce un vettore di due elemeti. immagini è una matrice NxM, dove N
% sono le immagini, eventualmente ridotte a vettore.

function [immagini,smiles, dim_comune] = collectImagesAndSmilesFromUrl ( path, ext, varargin )

    % Flag che stabilisce se convertire le immagini in scala di grigio.
    riduciInGray = 1;
    % Controlla se la dimensione delle immagini è la stessa.
    %   In caso contrario arresta l'esecuzione
    checkDimensions = 1;
    % Rende le mappe immagini un filotto (array colonna) prima di
    % immagazzinarle
    filotto = 1;
    % Effettua un ridimensionamento sulle immagini
    resize = 0;
    if numel(varargin) > 0
        resize = 1;
        resize_dim = varargin{1};
    end

    % Estraggo gli indirizzi delle immagini in path
    indirizzi = extractImgsPath( path, ext );
    %indirizzi = arrayShuffle( indirizzi );
    
    
    % Carico le immagini, in gray scale, in un array
    n = numel(indirizzi);
    immagini = [];
    for i = 1:n
        addr = indirizzi{i}; %addr = indirizzi{i};
        im1 = imread( addr );
        if riduciInGray == 1
            im1 = rgb2gray( im1 );
        end

        if i==1
            dim_comune = size(im1);
        end
        if checkDimensions == 1 & dim_comune ~= size(im1) ;
            error('Le immagini non hanno la stessa dimensione!');
        end
        dim_comune = size( im1 );
        if resize == 1 
            im1 = imresize( im1, [resize_dim(1), resize_dim(2)] );
        end
        if filotto == 1
            im1 = im1(:);
        end

        immagini = [ immagini, im1 ];
    end
    immagini = immagini';
    smiles = tellMeIfSmiles( indirizzi,ext );
end