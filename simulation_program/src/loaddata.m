function array =loaddata(filename,size)
fileID = fopen(filename,'r');
formatSpec = '%f %f';
array = fscanf(fileID,formatSpec,size);
fclose(fileID);
