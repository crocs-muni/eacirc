function [data_mean, setChangeFitness] = plot_scores(fileNameTemplate, fromIndex, toIndex, column, color)

% READ ALL DATA STREAMS
offset = 1;
filePath = [fileNameTemplate '[' num2str(fromIndex) ']_0_2'];
%filePath = [fileNameTemplate num2str(fromIndex)];
a = load(filePath);
numValues = length(a(:,column));

NUM_FILES = toIndex-fromIndex+1;
data(1:NUM_FILES, 1:numValues) = 0;
for index = fromIndex : toIndex
    %filePath = [fileNameTemplate num2str(index)];
    filePath = [fileNameTemplate '[' num2str(index) ']_0_2'];
    a = load(filePath);
    data(offset, :) = a(:,column);
    
%    plot(a(:,column), 'b');
%    hold on;
    
    offset = offset + 1;
end
% LAST USED INDEX
offset = offset - 1;

% COMPUTE AVERAGE
data_mean = data(1,:);
if (NUM_FILES > 1) 
    data_mean = mean(data); 
%    data_mean = max(data); 
%    data_mean = min(data); 
end

% OBTAIN VALUE JUST AFTER TEST SET CHANGE
setChangeFitness = data_mean(1 : 1000 : end);
sum(setChangeFitness)

% PLOT RESULT
hold on;
plot(data_mean, color);
