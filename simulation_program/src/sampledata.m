global simulation_data_path
startup()
simdata=fullfile(simulation_data_path,'output');
n_epochs =1000;
sample_rate = 100; % in Hz
length_epochs = 1110; % in ms
% each source will get the same type of activation: brown coloured noise
epochs     = struct('n', n_epochs, 'srate', 100, 'length', length_epochs);
epochs_noise = struct('n',n_epochs,'srate',100,'length', length_epochs);

% This function was originally used to randomly designate the noise source
% locations, outputting 20 sourceindexes which lie at least 2.5 cm apart.
% noise_locs = lf_get_source_spaced(leadfield,20,25);
% However, the line below just features the location numbers for one of
% those random sets and makes sure that in all simulations the noise
% sources are in the same locations.
noise_locs = [1321 996 1238	1520 316 130 1549 1974 558 563 306 310 1501	1587 1058 673 48 1990 1666 412];
channels = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3',...
    'P4', 'O1', 'O2', 'F7','F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2',...
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz'};

% Standard Deviations of erp target sources
% See Section 4.1.3 - Simulation parameters
stddevs=[0,0.2,0.4,0.6,0.8,1];

% Parameters characterizing the noises
% See Section 4.1.3 - Noise simulation
noise_params=[5 10 20; % number of sources
    1 10 100]; % Maximum source amplitude
    
if ~exist ('leadfield','var')
    leadfield   = lf_generate_fromfieldtrip('labels', channels);
end
%sourceIdx = lf_get_source_nearest(leadfield,[0,0,0]);

% Uncomment to save simulated artifact data
% For each artifact type there are 8 folders named X_0 ... X_7.
% The X is replaced by the artifact numbers: 
% 1:'Press feet',
% 2:'Lift tongue', 
% 4:'Clench teeth', 
% 6:'Push breath', 
% 7:'Wrinkle nose',
% 8:'Swallow',
savenoisedata(noise_params,leadfield,epochs_noise,simdata,channels,noise_locs);

% Uncomment to save simulated erp data
% Folders for the different sigma parameters are labelled stddevX.X, e.g
% stddev0.6
saveerpdata(stddevs,noise_params,leadfield,epochs_noise,simdata,channels,noise_locs);

