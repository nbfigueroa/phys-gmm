function [S, true_labels] = loadSimilarityConfMatrix(data_path, dataset)

switch dataset
    case 'YouTube'
        data = strcat(data_path, 'YouTube-Sim-Matrix.mat');
        load(data)
end
end