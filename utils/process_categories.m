function [results] = process_categories(fileNameTemplate, numFiles, column)

results(1:numFiles / 2 - 1, 1:numFiles / 2 - 1) = 0;
for index = 1 : numFiles / 2 - 1
    data1 = plot_scores(fileNameTemplate, index, index + numFiles / 2, column, 'r');
    for index2 = 1 : numFiles / 2 - 1
        data2 = plot_scores(fileNameTemplate, index + 1, index + 1 + numFiles / 2, column, 'r');
        diference = data1 - data2;
    
        results(index, index2) = sum(diference);
    end
end

