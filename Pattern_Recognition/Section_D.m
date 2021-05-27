clc;
clear;
close all;
load('F0_PVT_50.mat');

%% Section D1 Hierarchy algorithm with Standardised Data
Stan_Data = zscore(Data')';
species = { 'A';'A';'A';'A';'A';'A';'A';'A';'A';'A'
            'B';'B';'B';'B';'B';'B';'B';'B';'B';'B'
            'C';'C';'C';'C';'C';'C';'C';'C';'C';'C'
            'F';'F';'F';'F';'F';'F';'F';'F';'F';'F'
            'K';'K';'K';'K';'K';'K';'K';'K';'K';'K'
            'S';'S';'S';'S';'S';'S';'S';'S';'S';'S'
            };

tree = linkage(Stan_Data','average');

D_eucli = pdist(Stan_Data','euclidean');
leafOrder_1 = optimalleaforder(tree,D_eucli);
D_Man = pdist(Stan_Data','cityblock');
leafOrder_2 = optimalleaforder(tree,D_Man);

figure();
[H_1,T_1,outperm_1] = dendrogram(tree,60,'Reorder',leafOrder_1,'Labels',species);
xlabel('A for acrylic, B for black foam, C for car sponge, F for flour sack, K for kitchen sponge, S for steel vase')
%title('Hierarchical Clustering with Euclidean using standardised data')

figure();
[H_2,T_2,outperm_2] = dendrogram(tree,60,'Reorder',leafOrder_2,'Labels',species);
xlabel('A for acrylic, B for black foam, C for car sponge, F for flour sack, K for kitchen sponge, S for steel vase')
%title('Hierarchical Clustering with Manhattan using standardised data')

%% Section D1 Hierarchy algorithm with Zero-cetered Data
centered_data = zeros(3,60);
for i = 1:3
centered_Data(i,:) =  mean(Data(i,:)) - Data(i,:);
end

tree = linkage(centered_Data','average');

D_eucli = pdist(centered_Data','euclidean');
leafOrder_1 = optimalleaforder(tree,D_eucli);
D_Man = pdist(centered_Data','cityblock');
leafOrder_2 = optimalleaforder(tree,D_Man);

figure();
[H_1,T_1,outperm_1] = dendrogram(tree,60,'Reorder',leafOrder_1,'Labels',species);
xlabel('A for acrylic, B for black foam, C for car sponge, F for flour sack, K for kitchen sponge, S for steel vase')
%title('Hierarchical Clustering with Euclidean using zero-cetered data')

figure();
[H_2,T_2,outperm_2] = dendrogram(tree,60,'Reorder',leafOrder_2,'Labels',species);
xlabel('A for acrylic, B for black foam, C for car sponge, F for flour sack, K for kitchen sponge, S for steel vase')
%title('Hierarchical Clustering with Manhattan using zero-cetered data')

%% Section D1 K-means algorithm with Standardised Data
Stan_Data = zscore(Data')';
%train k-means with sqeuclidean distance metric
[idx_euc,C_euc] = kmeans(Stan_Data',6,'Distance','sqeuclidean');
%train k-means with cityblock (Manhattan) distance metric
[idx_man,C_man] = kmeans(Stan_Data',6,'Distance','cityblock');

figure;
for i = 1:10:51
    scatter3(Stan_Data(1,i:i+9),Stan_Data(2,i:i+9),Stan_Data(3,i:i+9),'filled');
    hold on;
end

plot3(C_euc(:,1),C_euc(:,2),C_euc(:,3),'kx','LineWidth',2)
hold off;
%title('K-Means Clustering with Euclidean using standardised data');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase','Cluster Centroid')

figure;
for i = 1:10:51
    scatter3(Stan_Data(1,i:i+9),Stan_Data(2,i:i+9),Stan_Data(3,i:i+9),'filled');
    hold on;
end

plot3(C_man(:,1),C_man(:,2),C_man(:,3),'kx','LineWidth',2)
hold off;
%title('K-Means Clustering with Manhattan using standardised data');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase','Cluster Centroid')

%% Section D1 K-means algorithm with centered Data
centered_data = zeros(3,60);
for i = 1:3
centered_Data(i,:) =  mean(Data(i,:)) - Data(i,:);
end
%train k-means with sqeuclidean distance metric
[idx_euc,C_euc] = kmeans(centered_Data',6,'Distance','sqeuclidean');
%train k-means with cityblock (Manhattan) distance metric
[idx_man,C_man] = kmeans(centered_Data',6,'Distance','cityblock');

figure;
for i = 1:10:51
    scatter3(centered_Data(1,i:i+9),centered_Data(2,i:i+9),centered_Data(3,i:i+9),'filled');
    hold on;
end

plot3(C_euc(:,1),C_euc(:,2),C_euc(:,3),'kx','LineWidth',2)
hold off;
%title('K-Means Clustering with Euclidean using centered data');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase','Cluster Centroid')

figure;
for i = 1:10:51
    scatter3(centered_Data(1,i:i+9),centered_Data(2,i:i+9),centered_Data(3,i:i+9),'filled');
    hold on;
end

plot3(C_man(:,1),C_man(:,2),C_man(:,3),'kx','LineWidth',2)
hold off;
%title('K-Means Clustering with Manhattan using centered data');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase','Cluster Centroid')

%% Section D2 Bagging with Standardised Data
load ('F0_Electrodes_50.mat');
stan_data=zscore(Data');
Covariance=cov(stan_data);
[Eigenvectors,Eigenvalues]=eig(Covariance);
[dummy,order] = sort(diag(-Eigenvalues));
Feature_Vector = Eigenvectors(:,order);

projected_scores = stan_data * Feature_Vector;

first_observation = projected_scores(:,1);
second_observation = projected_scores(:,2);
third_observation = projected_scores(:,3);

figure();
for i = 1:10:51
    scatter3(first_observation(i:i+9),second_observation(i:i+9),...
    third_observation(i:i+9),'filled')
    hold on
end
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase','FontSize',10);
grid on
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
%title('Electrode data projected onto the largest three PCs using standardised data')

dataset=[first_observation,second_observation,third_observation];
species={'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';
    'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';
    'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge'
    'flour_sack';'flour_sack';'flour_sack';'flour_sack';'flour_sack';'flour_sack';'flour_sack';'flour_sack';'flour_sack';'flour_sack';
    'kitchen_sponge';'kitchen_sponge';'kitchen_sponge';'kitchen_sponge';'kitchen_sponge';'kitchen_sponge';'kitchen_sponge';'kitchen_sponge';'kitchen_sponge';'kitchen_sponge';
    'steel_vase';'steel_vase';'steel_vase';'steel_vase';'steel_vase';'steel_vase';'steel_vase';'steel_vase';'steel_vase';'steel_vase'};

% 60 / 40 split for Training / Test data.
[m,n] = size(dataset);
P = 0.6; % training_percentage
rng(2000) % control random number generator with seed
idx = randperm(m);  % returns a row vector containing a random permutation of the integers from 1 to n without repeating elements
training_dataset = dataset(idx(1:round(P*m)),:);
testing_dataset = dataset(idx(round(P*m)+1:end),:);
training_species = species(idx(1:round(P*m)),:);
testing_species = species(idx(round(P*m)+1:end),:);

TreeNum = 30;
Mdl = TreeBagger(TreeNum,training_dataset,training_species,'OOBPrediction','On','Method','classification')
view(Mdl.Trees{1},'Mode','graph')
view(Mdl.Trees{2},'Mode','graph')

figure;
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble)
xlabel('Number of grown trees');
ylabel('Out-of-bag classification error');
%title('Grown trees using standardised data');

predicted_species = Mdl.predict(testing_dataset);

% Known groups, Predicted groups
Confusion_matrix = confusionmat(testing_species,predicted_species)
figure();
ldaResubCM = confusionchart(testing_species,predicted_species);
% Accuracy
Accuracy = sum (diag(Confusion_matrix)) / sum (Confusion_matrix(:)) *100

%% Section D2 Bagging with Zero-centered Data
centered_data = zeros(3,60);
for i = 1:3
centered_data(i,:) =  mean(Data(i,:)) - Data(i,:);
end
Covariance=cov(centered_data');
[Eigenvectors,Eigenvalues]=eig(Covariance);
[dummy,order] = sort(diag(-Eigenvalues));
Feature_Vector = Eigenvectors(:,order);

projected_scores = centered_data' * Feature_Vector;

first_observation = projected_scores(:,1);
second_observation = projected_scores(:,2);
third_observation = projected_scores(:,3);

figure();
for i = 1:10:51
    scatter3(first_observation(i:i+9),second_observation(i:i+9),...
    third_observation(i:i+9),'filled')
    hold on
end
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase','FontSize',10);
grid on
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
%title('Electrode data projected onto the largest three PCs using zero-centered data')


dataset=[first_observation,second_observation,third_observation];

% 60 / 40 split for Training / Test data.
[m,n] = size(dataset);
P = 0.6; % training_percentage
rng(2000) % control random number generator with seed
idx = randperm(m);  % returns a row vector containing a random permutation of the integers from 1 to n without repeating elements
training_dataset = dataset(idx(1:round(P*m)),:);
testing_dataset = dataset(idx(round(P*m)+1:end),:);
training_species = species(idx(1:round(P*m)),:);
testing_species = species(idx(round(P*m)+1:end),:);

TreeNum = 30;
Mdl = TreeBagger(TreeNum,training_dataset,training_species,'OOBPrediction','On','Method','classification')
view(Mdl.Trees{1},'Mode','graph')
view(Mdl.Trees{2},'Mode','graph')

figure;
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble)
xlabel('Number of grown trees');
ylabel('Out-of-bag classification error');
%title('Grown trees using zero-centered data');

predicted_species = Mdl.predict(testing_dataset);

% Known groups, Predicted groups
Confusion_matrix = confusionmat(testing_species,predicted_species)
figure();
ldaResubCM = confusionchart(testing_species,predicted_species);
% Accuracy
Accuracy = sum (diag(Confusion_matrix)) / sum (Confusion_matrix(:)) *100