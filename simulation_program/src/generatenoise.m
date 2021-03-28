function noise_components=generatenoise(leadfield,n_sources,noise_locs,...
                                        varargin)
    p=inputParser;
    addRequired(p,'leadfield');
    addRequired(p,'n_sources');
    addParameter(p,'amplitude',0,@isnumeric);
    addParameter(p,'amplitude_stddev',0,@isnumeric);
    
    parse(p,leadfield,n_sources,varargin{:});
    amplitude_stddev=p.Results.amplitude_stddev;
    amplitude=p.Results.amplitude;
    
    noise = struct('type','noise','color','pink',...
                'amplitude',amplitude,'amplitudeDv',amplitude_stddev);


    for i = 1:n_sources
        noise_components(i)=utl_create_component(noise_locs(i),noise,leadfield);
        orientation = rand(1,3)*2-1;
        noise_components(i).orientation = orientation/norm(orientation);
    end
    
end