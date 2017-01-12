function  indirizzi  = extractImgsPath ( path, ext )
%% Esplora il path che gli si passa n argomento, collezionando gli indirizzi
%% delle immagini, di estensione ext, in un cell array.

%url = '../img/smiles'
%ext = 'jpg';

% Estraggo gli indirizzi delle immagini contenute nella directory path %
p = dir( path );
p = struct2cell(p);
pos = regexp( p(1,:), ext );
indirizzi = {};
n = numel(pos);
c = 1;
for i=1:n
    if pos{i} >= length(ext) ;
        str = strcat(path,'/',p(1,i));
        indirizzi{c} = str{1};
        c = c +1;
    end
end

end