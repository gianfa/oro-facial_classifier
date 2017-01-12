function output = tellMeIfSmiles( input, ext )
%Legge un array di path di files e, in base all'ultima lettera, prima
%dell'estensione del file, attribuisce un carattere, in base ad una condizione
%data.
% input è un array di path di files
% ext è un char, estensione dei files nell'array


c1 = 'a';
c2 = 'b';
co1 = 0;
co2 = 1;

ne = numel(ext);
n = numel( input );
output = zeros(1,n);
for i = 1:n
    el = input(i);
    el = cast( el, 'char' );
    el = el( numel(el)- ne - 1 );
    if el == c1
        output(i) = co1;
    elseif el == c2
        output(i) = co2;
    else
        display('errore');
    end
end

