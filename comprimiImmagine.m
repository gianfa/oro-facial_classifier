%% Comprime un'immagine tramite una svd
% A � l'immagine. Pu� essere una matrice o un vettore.
% k � il numero di autovettori da mantenere dopo la trasformazione
% Accetta le dimensioni di riferimento, per A vettore (in varargin).
% out � il vettore di immagini gi� trasformate, visualizzabili con imshow(
% uint8(out).
% autovettori � il vettore di autovettori.
% trasf � il vettore delle immagini trasformate.

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
    
    
    