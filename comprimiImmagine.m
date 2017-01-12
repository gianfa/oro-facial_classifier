%% Comprime un'immagine tramite una svd
% A è l'immagine. Può essere una matrice o un vettore.
% k è il numero di autovettori da mantenere dopo la trasformazione
% Accetta le dimensioni di riferimento, per A vettore (in varargin).
% out è il vettore di immagini già trasformate, visualizzabili con imshow(
% uint8(out).
% autovettori è il vettore di autovettori.
% trasf è il vettore delle immagini trasformate.

function [ out, U, S, V, autovettori, trasf ] = comprimiImmagine ( A, k, varargin )
        if numel( varargin ) > 0
            dim = varargin{1};
            A = reshape( A, dim(1), dim(2) );
        end
        A = double(A);
        %A = A(:,:,1);
        [U,S,V] = svd(A);
        autovettori = U(:,1:k);
        trasf = S(1:k,1:k)*V(:,1:k)';
        out = autovettori*trasf;
        %imshow(uint8(out));
    
    
    