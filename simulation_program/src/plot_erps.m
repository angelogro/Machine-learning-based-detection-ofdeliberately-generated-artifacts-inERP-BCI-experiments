function plot_erps(data,channels,plotchannels,epochs,...
                    varargin)
tempdata=data.data
for chan=plotchannels
    findChan = cellfun(@(x)isequal(x,chan),channels);
    [row,col] = find(findChan);
    data.data=tempdata(col,:,1);
    plot_signal_fromclass(data,epochs);
    set(gca, 'YDir', 'reverse')
end

end