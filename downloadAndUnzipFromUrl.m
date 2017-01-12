function [filenames] = downloadAndUnzipFromUrl( sourceUrl )
% Download a zip file from a url and unzip it in a new folder
% outputdir :  mkdir('db');
% sourceUrl = ['http://fei.edu.br/~cet/frontalimages_manuallyaligned_part1.zip'];

name = 'db'; %name of the new folder where unzi files

outputdir =  mkdir(name);
fullURL = sourceUrl;
filename = strcat( num2str(int8( rand(1,8)*9 ) ),'samples.zip' );
filename = strrep( filename, ' ', '' );
urlwrite(fullURL,filename,'Timeout',5);
filenames = unzip(filename,name);

delete(filename);
