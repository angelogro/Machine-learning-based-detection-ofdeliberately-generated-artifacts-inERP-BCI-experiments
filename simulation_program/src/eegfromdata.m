

% each source will get the same type of activation: brown coloured noise
config      = struct('n', 100, 'srate', 1000, 'length', 1000);

% obtaining a 64-channel lead field from ICBM-NY
leadfield   = lf_generate_fromnyhead('montage', 'S64');

% obtaining 64 random source locations, at least 2.5 cm apart
sourcelocs  = lf_get_source_spaced(leadfield, 64, 25);

% each source will get the same type of activation: brown coloured noise
signal      = struct('peakLatency', 200, 'peakWidth', 100, ...
               'peakAmplitude', 1);
signal = utl_check_class(signal, 'type', 'erp');

% packing source locations and activations together into components
components  = utl_create_component(sourcelocs, signal, leadfield);

% simulating data
data        = generate_scalpdata(components, leadfield, config);

% converting to EEGLAB dataset format, adding ICA weights
EEG         = utl_create_eeglabdataset(data, config, leadfield);
EEG         = utl_add_icaweights_toeeglabdataset(EEG, components, leadfield);

