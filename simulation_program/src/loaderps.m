function [components,sourceIdx]=loaderps(filename,leadfield,...
                                        varargin)
    p=inputParser;
    addRequired(p,'filename',@isstr);
    addRequired(p,'leadfield');
    addParameter(p,'amplitude_stddev',0,@isnumeric);
    addParameter(p,'latency_stddev',0,@isnumeric);
    addParameter(p,'pulsewidth_stddev',0,@isnumeric);
    addParameter(p,'location_stddev',0,@isnumeric);
    addParameter(p,'orientation_stddev',0,@isnumeric);
    
    ampl_norm_factor=4000; %Obtained with signal comparison to paper plot
    
    parse(p, filename,leadfield,varargin{:});
    amplitude_stddev=p.Results.amplitude_stddev;
    latency_stddev=p.Results.latency_stddev;
    pulsewidth_stddev=p.Results.pulsewidth_stddev;
    location_stddev=p.Results.location_stddev;
    orientation_stddev=p.Results.orientation_stddev;
                                    
    T=readtable(filename);
    [n_rows,n_columns] = size(T);

    for i = 1:n_rows
        erp = struct('type', 'erp', ...
        'peakLatency',T.Latency_mean_(i), ...
        'peakLatencyDv',5*latency_stddev*(T.Latency_max_(i)-T.Latency_min_(i)),...
        'peakWidth', T.Pulsewidth(i),...
        'peakWidthDv',T.Pulsewidth(i)*(pulsewidth_stddev),...
        'peakAmplitude', ampl_norm_factor*T.Amplitude_mean_(i),...
        'peakAmplitudeDv',5*ampl_norm_factor*(amplitude_stddev)*(T.Amplitude_max_(i)-T.Amplitude_min_(i)));
        location = [T.x(i)*(1+normrnd(0,location_stddev)),...
            T.y(i)*(1+normrnd(0,location_stddev)),...
            T.z(i)*(1+normrnd(0,location_stddev))];
        orientation = [T.OrientationX(i),...
            T.OrientationY(i),...
            T.OrientationZ(i)];
        sourceIdx(i) = lf_get_source_nearest(leadfield,location);
        components(i)=utl_create_component(sourceIdx(i),erp,leadfield);
        components(i).orientation = orientation/norm(orientation);
        components(i).orientationDv = [3*orientation_stddev,3*orientation_stddev,3*orientation_stddev];
    end
    
end