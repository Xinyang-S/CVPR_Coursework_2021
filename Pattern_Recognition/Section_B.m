clc;
close all;
clear;
load ('F0_PVT_50.mat');

%% Section B1 Use Standardised Data to calculate covariance and eigenvectors and eigenvalue
stan_data = zscore(Data')'; % standisation
cov_matrix = cov(stan_data') % Calculate covariance matrix
% calculate Covariance and sorted eigenvectors and eigenvalue
[eig_vector,eig_val] = eig(cov_matrix); 
eig_vector = (rot90(eig_vector))'
eig_val = diag(rot90(rot90(eig_val)))'

% Assigned to principle components
three_dimensions = zeros(3,60);
for k =1:1:60
    three_dimensions(1,k) = (eig_vector(:,1)')*(stan_data(:,k));
    three_dimensions(2,k) = (eig_vector(:,2)')*(stan_data(:,k));
    three_dimensions(3,k) = (eig_vector(:,3)')*(stan_data(:,k));
end

% plot standardised data with principle components (three dimensions)
figure;
for counter = 1:10:51
    scatter3(three_dimensions(1,counter:counter+9),three_dimensions(2,counter:counter+9),three_dimensions(3,counter:counter+9),'filled');
    hold on;
end

plot3([0,0],[1,0],[0,0],'LineWidth',1.5,'Color','b');
plot3([1,0],[0,0],[0,0],'LineWidth',1.5,'Color','r');
plot3([0,0],[0,0],[1,0],'LineWidth',1.5,'Color','g');

xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');

% plot standardised data with principle components (two dimensions)
figure;
for counter = 1:10:51
    scatter(three_dimensions(1,counter:counter+9),three_dimensions(2,counter:counter+9),'filled');
    hold on;
end

plot([0,0],[0,1],'LineWidth',1.5,'Color','g');
plot([1,0],[0,0],'LineWidth',1.5,'Color','r');

xlabel('PC1');
ylabel('PC2');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');
%title('2D Standardised data with the Principal components displayed');

figure;
ax_1 =zeros(1,60);
ax_2 =zeros(1,60)+0.5;
ax_3 =zeros(1,60)+1;
for counter = 1:10:51
    scatter(three_dimensions(1,counter:counter+9),ax_1(1,counter:counter+9));
    scatter(three_dimensions(2,counter:counter+9),ax_2(1,counter:counter+9));
    scatter(three_dimensions(3,counter:counter+9),ax_3(1,counter:counter+9));
    hold on;
end
set(gca,'ytick',[]);
ylim([-1 2])
ylabel('PC1       PC2       PC3');
hold on
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');
%title('1D Standardised data lines');

%% Section B1 Use Zero-centered Data to calculate covariance and eigenvectors and eigenvalue
centered_data = zeros(3,60);
% calculate zero-centred data
for para = 1:1:3
    centered_data(para,:) =  mean(Data(para,:))- Data(para,:);
end

cov_matrix = cov(centered_data');
[eig_vector,eig_val] = eig(cov_matrix);
eig_vector = (rot90(eig_vector))';
eig_val = diag(rot90(rot90(eig_val)))';


three_dimensions = zeros(3,60);
for k =1:1:60
    three_dimensions(1,k) = (eig_vector(:,1)')*(centered_data(:,k));
    three_dimensions(2,k) = (eig_vector(:,2)')*(centered_data(:,k));
    three_dimensions(3,k) = (eig_vector(:,3)')*(centered_data(:,k));
end

figure;
for counter = 1:10:51
    scatter3(three_dimensions(1,counter:counter+9),three_dimensions(2,counter:counter+9),three_dimensions(3,counter:counter+9),'filled');
    hold on;
end

plot3([0,0],[40,0],[0,0],'LineWidth',1.5,'Color','b');
plot3([100,0],[0,0],[0,0],'LineWidth',1.5,'Color','r');
plot3([0,0],[0,0],[15,0],'LineWidth',1.5,'Color','g');

xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');

three_dimensions = zeros(3,60);
for k =1:1:60
    three_dimensions(1,k) = (eig_vector(:,1)')*(centered_data(:,k));
    three_dimensions(2,k) = (eig_vector(:,2)')*(centered_data(:,k));
    three_dimensions(3,k) = (eig_vector(:,3)')*(centered_data(:,k));
    hold on;
end

figure;
for counter = 1:10:51
    scatter(three_dimensions(1,counter:counter+9),three_dimensions(2,counter:counter+9),'filled');
    hold on;
end

plot([0,0],[0,1*50],'LineWidth',1.5,'Color','g');
plot([1*150,0],[0,0],'LineWidth',1.5,'Color','r');

xlabel('PC1');
ylabel('PC2');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');


figure;
ax_1 =zeros(1,60);
ax_2 =zeros(1,60)+0.5;
ax_3 =zeros(1,60)+1;
for counter = 1:10:51
    scatter(three_dimensions(1,counter:counter+9),ax_1(1,counter:counter+9));
    scatter(three_dimensions(2,counter:counter+9),ax_2(1,counter:counter+9));
    scatter(three_dimensions(3,counter:counter+9),ax_3(1,counter:counter+9));
    hold on;
end
set(gca,'ytick',[]);
ylim([-1 2])
ylabel('PC1       PC2       PC3');
hold on
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');
%title('1D Zero-centered data lines');


%% Section B2 Use Standardised electrodes Data 
load ('F0_Electrodes_50.mat');
stan_data = zscore(Data')'; %Standisation
cov_matrix = cov(stan_data'); %Calculate convariance
% calculate Covariance and sorted eigenvectors and eigenvalue
[eig_vector,eig_val] = eig(cov_matrix);
eig_vector = (rot90(eig_vector))';
eig_val = diag(rot90(rot90(eig_val)))';

% plot scree figure
figure();
plot(1:19,eig_val,'b--o');
ylabel('Variance');
xlabel('Number of components');


three_main_dimensions = zeros(3,60);
for k =1:1:60
    three_main_dimensions(1,k) = (eig_vector(:,1)')*(stan_data(:,k));
    three_main_dimensions(2,k) = (eig_vector(:,2)')*(stan_data(:,k));
    three_main_dimensions(3,k) = (eig_vector(:,3)')*(stan_data(:,k));
    hold on;
end

stan_data = zeros(19,60);
for para = 1:1:19
    stan_data(para,:) =  mean(Data(para,:))- Data(para,:);
end

% plot three-dimensional standardised eletrode data assigned to three most principle components 
figure;
for counter = 1:10:51
    scatter3(three_main_dimensions(1,counter:counter+9),three_main_dimensions(2,counter:counter+9),three_main_dimensions(3,counter:counter+9),'filled');
    hold on;
end

plot3([0,0],[3,0],[0,0],'LineWidth',1.5,'Color','b');
plot3([3,0],[0,0],[0,0],'LineWidth',1.5,'Color','r');
plot3([0,0],[0,0],[1.5,0],'LineWidth',1.5,'Color','g');

xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');
%title('Three principal components with largest variance using standardised data');

%% Section B2 Use Zero-centered electrodes Data
centered_data = zeros(19,60);
for para = 1:1:19
    centered_data(para,:) =  mean(Data(para,:))- Data(para,:);
end
cov_matrix = cov(centered_data');
[eig_vector,eig_val] = eig(cov_matrix);
eig_vector = (rot90(eig_vector))';
eig_val = diag(rot90(rot90(eig_val)))';
figure();
plot(1:19,eig_val,'b--o');
ylabel('Variance');
xlabel('Number of components');

three_main_dimensions = zeros(3,60);
for k =1:1:60
    three_main_dimensions(1,k) = (eig_vector(:,1)')*(centered_data(:,k));
    three_main_dimensions(2,k) = (eig_vector(:,2)')*(centered_data(:,k));
    three_main_dimensions(3,k) = (eig_vector(:,3)')*(centered_data(:,k));
    hold on;
end
%title('Scree plot using zero-centered data');
centered_data = zeros(19,60);
for para = 1:1:19
    centered_data(para,:) =  mean(Data(para,:))- Data(para,:);
end
figure;
for counter = 1:10:51
    scatter3(three_main_dimensions(1,counter:counter+9),three_main_dimensions(2,counter:counter+9),three_main_dimensions(3,counter:counter+9),'filled');
    hold on;
end

plot3([0,0],[100,0],[0,0],'LineWidth',1.5,'Color','b');
plot3([300,0],[0,0],[0,0],'LineWidth',1.5,'Color','r');
plot3([0,0],[0,0],[50,0],'LineWidth',1.5,'Color','g');

xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');
%title('Three principal components with largest variance using zero-centered data');


