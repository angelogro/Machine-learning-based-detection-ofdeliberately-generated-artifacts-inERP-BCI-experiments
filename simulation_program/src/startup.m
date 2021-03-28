function returnvalue=startup()
global simulation_data_path
[filepath,name,ext] = fileparts(mfilename('fullpath'));
parts = strsplit(filepath, filesep);
sim_path = strjoin(parts(1:end-1), filesep);
project_path = strjoin(parts(1:end-2), filesep);
data_path = fullfile(project_path,'data');
simulation_data_path = fullfile(data_path,'simulation_data');
simlib_path = fullfile(sim_path,'lib');
addpath(fullfile(simlib_path,'fieldtrip-20200607'));
addpath(project_path);
addpath(genpath(project_path));
mkdir(fullfile(simulation_data_path,'output'));
ft_defaults
clearvars;
returnvalue=true;