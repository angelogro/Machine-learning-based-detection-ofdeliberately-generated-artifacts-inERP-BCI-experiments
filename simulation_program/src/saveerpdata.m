function saveerpdata(erp_params,noise_params,leadfield,epochs_noise,savefolder,channels,noise_locs)

% Iterate through the erp parameter sigma (0,0.2,0.4,0.6,0.8,1)
% See 4.2 - Data Generation
for erp_param=erp_params

    % Source location and orientation:
    % https://www.jneurosci.org/content/jneuro/24/42/9353.full.pdf
    % Creates the different sources (20 in total) with different sigma.
    
% SECTION 4.1.3 - ERP Sources
    [erps,sourceIdxes] = loaderps('P300_generators.csv',leadfield,'amplitude_stddev',erp_param,...
        'latency_stddev',erp_param,'pulsewidth_stddev',erp_param,'orientation_stddev',erp_param);
    erp_data = generate_scalpdata(erps,leadfield,epochs_noise);
    
    % Iterate through the different noise types
% See 4.2 - Data Generation
    for noise_type=noise_params
        noise_components = generatenoise(leadfield,noise_type(1),noise_locs,'amplitude',5000*noise_type(2));
        noise_data = generate_scalpdata(noise_components,leadfield,epochs_noise);

        data        = struct();
        data.data=erp_data+noise_data;
        data.index = {'e',':'};
        data.amplitude = 1;
        data.amplitudeType= 'relative';

        data = utl_check_class(data,'type','data');
        
        % Uncomment to plot some of the data, as used in thesis
% See Figure 4.1
        % plot_erps(data,channels,["Cz"],epochs_noise);
        mkdir(fullfile(savefolder,strcat("stddev",string(erp_param))));
        writematrix(data.data,fullfile(savefolder,strcat("stddev",string(erp_param)),strcat(string(noise_type(1)),'.csv')));
    end

end