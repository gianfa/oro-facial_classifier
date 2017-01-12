%% Comprime un set di immagini
% set è una matrice NxM, dove le M sono le immagini ridotte a vettore
% colonna.
% k è il numero di autovalori che si vuol mantenere nella compressione.
% imgs_tr è una matrice FxI, con F features e I immagini.
% imgs_autovettori è la matrice degli autovettori.
% Per visualizzare una immagine del set compresso risultante basterà fare
% av = cell2mat( imgs_autovettori( n ));
% av_col = size( cell2mat( imgs_autovettori( n )),2);
% imshow (uint8( av * reshape( imgs_tr( :, n), av_col, numel( imgs_tr( :, n) )/av_col) )).
function [ imgs_tr, imgs_autovettori, imgs_out ] = comprimiSet ( set, k, varargin )
        dim = varargin{1};
        imgs_out = {};
        for n = 1:size( set,2 )
            cur = set(:,n);
            [ out, U, S, V, av, trasf ] = comprimiImmagine( cur, k, [dim(1), dim(2)] );
            if n == 1
                % creazione del contenitore di immagini trasformate
                imgs_tr = trasf(:);
                imgs_autovettori = { av };
            else 
                imgs_tr = [ imgs_tr, trasf(:) ];
                imgs_autovettori = [ imgs_autovettori, av ];
            end;
            imgs_out = [ imgs_out, out ];
        end
end