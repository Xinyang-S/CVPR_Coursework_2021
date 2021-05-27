clc;
clear;
close all;
load('F0_PVT_50.mat');
% Standardisation
F0_PVT_normalised = zscore(Data(:,11:30)')';
pressure = F0_PVT_normalised(1,:);
vibration = F0_PVT_normalised(2,:);
temp = F0_PVT_normalised(3,:);

%% Section_C_1_a
figure()
% Pressure-Vibration
subplot(3,1,1)
size = 30;

scatter(pressure(1:10),vibration(1:10),'filled','MarkerFaceColor','#D95319','SizeData',size)
hold on
scatter(pressure(11:20),vibration(11:20),'filled','MarkerFaceColor','#EDB120','SizeData',size)

% Apply LDA
Data_nor = F0_PVT_normalised;
species={'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';
     'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';'car_sponge';};

Data_nor_PV = Data_nor(1:2,:);  % for P,V datasets
LDA_PV = fitcdiscr(Data_nor_PV',species);    
[eigenvectors,eigenvalues]=eig(inv(LDA_PV.Sigma)*LDA_PV.BetweenSigma); %calcualte corresponding eigenvectors and values
plot([-eigenvectors(1,1)*2 eigenvectors(1,1)*2 ],...
    [-eigenvectors(2,1)*2  eigenvectors(2,1)*2 ],...
    'Color','#4DBEEE','lineWidth',1.5)   
hold on
% Calculate slope
LDA_slope = eigenvectors(2,1)/eigenvectors(1,1);
Discrimination_slope = -1/LDA_slope;
x = 10; 
y = Discrimination_slope*x;% perpendicular discrimination line 
plot([x -x],[y -y],'--','lineWidth',1.5)  
hold off
grid on
legend('black foam','car sponge','FontSize',10);
axis([-2 2 -2 2])
axis square
xlabel('Pressure')
ylabel('Vibration')



% Pressure-Temperature
subplot(3,1,2)

scatter(pressure(1:10),temp(1:10),'filled','MarkerFaceColor','#D95319','SizeData',size)
hold on
scatter(pressure(11:20),temp(11:20),'filled','MarkerFaceColor','#EDB120','SizeData',size)

% Apply LDA
Data_nor = F0_PVT_normalised;

Data_nor_PT = [Data_nor(1,:);Data_nor(3,:)];  % for P,T datasets
LDA_PT = fitcdiscr(Data_nor_PT',species);
[eigenvectors,eigenvalues]=eig(inv(LDA_PT.Sigma)*LDA_PT.BetweenSigma); %calcualte corresponding eigenvectors and values
plot([-eigenvectors(1,1)*2 eigenvectors(1,1)*2 ],...
    [-eigenvectors(2,1)*2  eigenvectors(2,1)*2 ],...
    'Color','#4DBEEE','lineWidth',1.5)   
hold on
% find slope
LDA_slope = eigenvectors(2,1)/eigenvectors(1,1);
Discrimination_slope = -1/LDA_slope;
x = 10; 
y = Discrimination_slope*x; % perpendicular discrimination line 
plot([x -x],[y -y],'--','lineWidth',1.5)  
hold off
grid on
legend('black foam','car sponge','FontSize',10);
axis([-2 2 -2 2])
axis square
xlabel('Pressure')
ylabel('Temperature')



% Temperature-Vibration
subplot(3,1,3)

scatter(temp(1:10),vibration(1:10),'filled','MarkerFaceColor','#D95319','SizeData',size)
hold on
scatter(temp(11:20),vibration(11:20),'filled','MarkerFaceColor','#EDB120','SizeData',size)

% Apply LDA
Data_nor = F0_PVT_normalised;

Data_nor_TV = [Data_nor(3,:);Data_nor(2,:)]; % for TV datasets
LDA_TV = fitcdiscr(Data_nor_TV',species);
[eigenvectors,eigenvalues]=eig(inv(LDA_TV.Sigma)*LDA_TV.BetweenSigma);%calcualte corresponding eigenvectors and values
plot([-eigenvectors(1,1)*2 eigenvectors(1,1)*2 ],...
    [-eigenvectors(2,1)*2  eigenvectors(2,1)*2 ],...
    'Color','#4DBEEE','lineWidth',1.5) 
hold on
% Calculate LDA line slope
LDA_slope = eigenvectors(2,1)/eigenvectors(1,1);
Discrimination_slope = -1/LDA_slope;
x = 10; 
y = Discrimination_slope*x; % perpendicular discrimination line
plot([x -x],[y -y],'--','lineWidth',1.5)  
hold off
grid on
legend('black foam','car sponge','FontSize',10);
axis([-2 2 -2 2])
axis square
xlabel('Temperature')
ylabel('Vibration')



%% Section_C_1_b
figure()
% 3D scatter plot the Data_nor points
scatter3(pressure(1:10),vibration(1:10),temp(1:10),'filled','MarkerFaceColor','#D95319','SizeData',size)
hold on
scatter3(pressure(11:20),vibration(11:20),temp(11:20),'filled','MarkerFaceColor','#EDB120','SizeData',size)

Data_nor_PVT = F0_PVT_normalised;
LDA_PVT = fitcdiscr(Data_nor_PVT',species,'DiscrimType','linear');

[Eigenvectors,Eigenvalues]=eig(inv(LDA_PVT.Sigma)*LDA_PVT.BetweenSigma);%calcualte corresponding eigenvectors and values
[Eigenvalues,index] = sort(diag(Eigenvalues),'descend');% sort the eigenvalues from big to small
Eigenvalues = Eigenvalues(index);
Feature_Vector = Eigenvectors(:,index);
Feature_Vector(:,3)=[]; 
% plot the feature vectors with hyperplane
X1 = [-Feature_Vector(1,1) Feature_Vector(1,1)]*2;
Y1 = [-Feature_Vector(2,1) Feature_Vector(2,1)]*2;
Z1 = [-Feature_Vector(3,1) Feature_Vector(3,1)]*2;
plot3(X1,Y1,Z1,'Color','#4DBEEE','lineWidth',1.5)
X2 = [-Feature_Vector(1,2) Feature_Vector(1,2)]*2;
Y2 = [-Feature_Vector(2,2) Feature_Vector(2,2)]*2;
Z2 = [-Feature_Vector(3,2) Feature_Vector(3,2)]*2;
plot3(X2,Y2,Z2,'lineWidth',1.5)
hold on
patch('XData',[-Feature_Vector(1,1:2)*10 Feature_Vector(1,1:2)*10],'YData',[-Feature_Vector(2,1:2)*10 Feature_Vector(2,1:2)*10],'ZData',[-Feature_Vector(3,1:2)*10 Feature_Vector(3,1:2)*10],'FaceAlpha',0.5)
hold on

hold off
grid on
legend('black foam','car sponge','FontSize',10);
axis([-2 2 -2 2 -2 2])
axis square
xlabel('Pressure')
ylabel('Vibration')
zlabel('Temperature')


Data_nor_PV = Data_nor(1:2,:);  
LDA_PV = fitcdiscr(Data_nor_PV',species);    
[eigenvectors,eigenvalues]=eig(inv(LDA_PV.Sigma)*LDA_PV.BetweenSigma);

figure
Data_nor_PV=Data_nor_PV';
Data_nor_1=Data_nor_PV*eigenvectors(:,1);
Data_nor_2=Data_nor_PV*eigenvectors(:,2);
LD1 = Data_nor_1(:,1);
Vertical= Data_nor_2(:,1);
step=1;
a=zeros(1,40);

scatter(LD1(1:10),a(1:10),'filled','MarkerFaceColor','#D95319','SizeData',size)
hold on
scatter(LD1(11:20),a(11:20),'filled','MarkerFaceColor','#EDB120','SizeData',size)

plot([-2 2],[0 0],'Color','#4DBEEE','lineWidth',1.5)
plot([0 0],[-2 2],'--','lineWidth',1.5)
axis([-2 2 -2 2 -2 2])
axis square
legend('black foam','car sponge','FontSize',10);


%% Section_C_1_d
% LDA between the two objects, black foam & acrylic
F0_PVT_normalised = zscore(Data(:,1:20)')'
pressure = F0_PVT_normalised(1,:);
vibration = F0_PVT_normalised(2,:);
temp = F0_PVT_normalised(3,:);

figure()
% ----------------------Pressure vs. Vibration------------------------
subplot(3,1,1)
% plot the Data_nor points
size = 30;

scatter(pressure(1:10),vibration(1:10),'filled','MarkerFaceColor','#0072BD','SizeData',size)
hold on
scatter(pressure(11:20),vibration(11:20),'filled','MarkerFaceColor','#D95319','SizeData',size)

% Apply LDA
Data_nor = F0_PVT_normalised;
species={'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';'acrylic';
        'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam';'black_foam'};

Data_nor_PV = Data_nor(1:2,:); 
LDA_PV = fitcdiscr(Data_nor_PV',species);    
[eigenvectors,eigenvalues]=eig(inv(LDA_PV.Sigma)*LDA_PV.BetweenSigma);
plot([-eigenvectors(1,1)*2 eigenvectors(1,1)*2 ],...
    [-eigenvectors(2,1)*2  eigenvectors(2,1)*2 ],...
    'Color','#4DBEEE','lineWidth',1.5)    
hold on
% find slope
LDA_slope = eigenvectors(2,1)/eigenvectors(1,1);
Discrimination_slope = -1/LDA_slope;
x = 10; y = Discrimination_slope*x; % perpendicular discrimination line
plot([x -x],[y -y],'--','lineWidth',1.5)  
hold off
grid on
legend('acrylic','black foam','FontSize',10);
axis([-2 2 -2 2])
axis square
xlabel('Pressure')
ylabel('Vibration')
%title('Spilted Data_nor by LDA line P-V')


% Pressure vs. Temperature
subplot(3,1,2)

scatter(pressure(1:10),temp(1:10),'filled','MarkerFaceColor','#0072BD','SizeData',size)
hold on
scatter(pressure(11:20),temp(11:20),'filled','MarkerFaceColor','#D95319','SizeData',size)

% Apply LDA
Data_nor = F0_PVT_normalised;

Data_nor_PT = [Data_nor(1,:);Data_nor(3,:)];  % for PT
LDA_PT = fitcdiscr(Data_nor_PT',species);
[eigenvectors,eigenvalues]=eig(inv(LDA_PT.Sigma)*LDA_PT.BetweenSigma);
plot([-eigenvectors(1,1)*2 eigenvectors(1,1)*2 ],...
    [-eigenvectors(2,1)*2  eigenvectors(2,1)*2 ],...
    'Color','#4DBEEE','lineWidth',1.5)   
hold on
% Calculate slope
LDA_slope = eigenvectors(2,1)/eigenvectors(1,1);
Discrimination_slope = -1/LDA_slope;
x = 10; 
y = Discrimination_slope*x; % perpendicular discrimination line
plot([x -x],[y -y],'--','lineWidth',1.5)  
hold off
grid on
legend('acrylic','black foam','FontSize',10);
axis([-2 2 -2 2])
axis square
xlabel('Pressure')
ylabel('Temperature')



% Temperature vs. Vibration
subplot(3,1,3)

scatter(temp(1:10),vibration(1:10),'filled','MarkerFaceColor','#0072BD','SizeData',size)
hold on
scatter(temp(11:20),vibration(11:20),'filled','MarkerFaceColor','#D95319','SizeData',size)

% Apply LDA
Data_nor = F0_PVT_normalised;

Data_nor_TV = [Data_nor(3,:);Data_nor(2,:)];  % for TV
LDA_TV = fitcdiscr(Data_nor_TV',species);
[eigenvectors,eigenvalues]=eig(inv(LDA_TV.Sigma)*LDA_TV.BetweenSigma);
plot([-eigenvectors(1,1)*2 eigenvectors(1,1)*2 ],...
    [-eigenvectors(2,1)*2  eigenvectors(2,1)*2 ],...
    'Color','#4DBEEE','lineWidth',1.5)   
hold on
% Calculate slope
LDA_slope = eigenvectors(2,1)/eigenvectors(1,1);
Discrimination_slope = -1/LDA_slope;
x = 10; 
y = Discrimination_slope*x;% perpendicular discrimination line
plot([x -x],[y -y],'--','lineWidth',1.5)  
hold off
grid on
legend('acrylic','black foam','FontSize',10);
axis([-2 2 -2 2])
axis square
xlabel('Temperature')
ylabel('Vibration')


figure()

scatter3(pressure(1:10),vibration(1:10),temp(1:10),'filled','MarkerFaceColor','#0072BD','SizeData',size)
hold on
scatter3(pressure(11:20),vibration(11:20),temp(11:20),'filled','MarkerFaceColor','#D95319','SizeData',size)

Data_nor_PVT = F0_PVT_normalised; 
LDA_PVT = fitcdiscr(Data_nor_PVT',species,'DiscrimType','linear');

[Eigenvectors,Eigenvalues]=eig(inv(LDA_PVT.Sigma)*LDA_PVT.BetweenSigma);
[Eigenvalues,index] = sort(diag(Eigenvalues),'descend');
Eigenvalues = Eigenvalues(index);
Feature_Vector = Eigenvectors(:,index);
Feature_Vector(:,3)=[]; % delete the last colume with lowest eigenvalue
X1 = [-Feature_Vector(1,1) Feature_Vector(1,1)]*2;
Y1 = [-Feature_Vector(2,1) Feature_Vector(2,1)]*2;
Z1 = [-Feature_Vector(3,1) Feature_Vector(3,1)]*2;
plot3(X1,Y1,Z1,'Color','#4DBEEE','lineWidth',1.5)
X2 = [-Feature_Vector(1,2) Feature_Vector(1,2)]*2;
Y2 = [-Feature_Vector(2,2) Feature_Vector(2,2)]*2;
Z2 = [-Feature_Vector(3,2) Feature_Vector(3,2)]*2;
plot3(X2,Y2,Z2,'lineWidth',1.5)
hold on
patch('XData',[-Feature_Vector(1,1:2)*10 Feature_Vector(1,1:2)*10],'YData',[-Feature_Vector(2,1:2)*10 Feature_Vector(2,1:2)*10],'ZData',[-Feature_Vector(3,1:2)*10 Feature_Vector(3,1:2)*10],'FaceAlpha',0.5)
hold on

hold off
grid on
legend('acrylic','black foam','FontSize',10);
axis([-2 2 -2 2 -2 2])
axis square
xlabel('Pressure')
ylabel('Vibration')
zlabel('Temperature')


Data_nor_PV = Data_nor(1:2,:);  % for PV
LDA_PV = fitcdiscr(Data_nor_PV',species);    
[eigenvectors,eigenvalues]=eig(inv(LDA_PV.Sigma)*LDA_PV.BetweenSigma);

figure
Data_nor_PV=Data_nor_PV'
Data_nor_1=Data_nor_PV*eigenvectors(:,1)
Data_nor_2=Data_nor_PV*eigenvectors(:,2)
LD1 = Data_nor_1(:,1);
Vertical= Data_nor_2(:,1);
step=1;
a=zeros(1,40);

scatter(LD1(1:10),a(1:10),'filled','MarkerFaceColor','#0072BD','SizeData',size)
hold on
scatter(LD1(11:20),a(11:20),'filled','MarkerFaceColor','#D95319','SizeData',size)

plot([-2 2],[0 0],'Color','#4DBEEE','lineWidth',1.5)
plot([0 0],[-2 2],'--','lineWidth',1.5)
axis([-2 2 -2 2 -2 2])
axis square
legend('acrylic','black foam','FontSize',10);

