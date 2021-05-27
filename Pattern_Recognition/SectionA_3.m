clear;
close all;
load('F0_PVT_50.mat');
figure;
scatter3(Data(1,1:10),Data(2,1:10),Data(3,1:10),'filled');
hold on;
grid on;
scatter3(Data(1,11:20),Data(2,11:20),Data(3,11:20),'filled');
scatter3(Data(1,21:30),Data(2,21:30),Data(3,21:30),'filled');
scatter3(Data(1,31:40),Data(2,31:40),Data(3,31:40),'filled');
scatter3(Data(1,41:50),Data(2,41:50),Data(3,41:50),'filled');
scatter3(Data(1,51:60),Data(2,51:60),Data(3,51:60),'filled');
xlabel('Pressure');
ylabel('Vibration');
zlabel('Temperature');
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');

figure;
a = ones(1,60);
for i = 1:10:51
    scatter(Data(1,i:i+9),a(1,i:i+9));
    hold on
end
set(gca,'ytick',[],'yticklabel',[])
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');

figure;
a = ones(1,60);
for i = 1:10:51
    scatter(Data(2,i:i+9),a(1,i:i+9));
    hold on
end
set(gca,'ytick',[],'yticklabel',[])
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');

figure;
a = ones(1,60);
for i = 1:10:51
    scatter(Data(3,i:i+9),a(1,i:i+9));
    hold on
end
set(gca,'ytick',[],'yticklabel',[])
legend('acrylic','black foam','car sponge','flour sack','kitchen sponge','steel vase');


