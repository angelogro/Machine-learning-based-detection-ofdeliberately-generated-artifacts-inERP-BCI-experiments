function savenoisedata(noise_params,leadfield,epochs_noise,savefolder,channels,noise_locs)
loadfolder='/home/angelo/Daten/Master_Arbeit/Master_Arbeit/Code/SimulationData/signal_data';
for noise_type=noise_params
    noise_components = generatenoise(leadfield,noise_type(1),noise_locs,'amplitude',5000*noise_type(2));

    %erp_data = generate_scalpdata(erps,leadfield,epochs_noise);
    
    %compl_data = erp_data+noise_data;

    % simulating data (Artifacts)
    folder=dir(loadfolder);
    [s1 s2]=size(folder);
    for artnum=['1','2','4','6','7','8']
        data        = struct();
        data.index = {'e',':'};
        data.amplitude = 1;
        data.amplitudeType= 'relative';
        
        for ele=1:s1
            name=string(folder(ele).name);
            if name.startsWith(artnum) & name.strlength()>2
                data.data = loaddata(fullfile(loadfolder,name),[length(channels) Inf]);
                noise_data = generate_scalpdata(noise_components,leadfield,epochs_noise);
                data.data=data.data+noise_data;
                mkdir(savefolder,name);

                data = utl_check_class(data,'type','data');
                writematrix(data.data,fullfile(savefolder,name,strcat(string(noise_type(1)),'.csv')));
            end
        end
    end
    
    %plot_erps(data,channels,["Cz"],epochs)
end