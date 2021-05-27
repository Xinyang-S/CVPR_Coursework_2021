%% Section A1
%The hold procedure seems like starting from about 35 of 1000. 
clc;
close all;
%% PVT
figure;
subplot(3,1,1);
plot(F0pdc, 'Displayname', 'Pressure');
%title('PVT');
xlabel('Pressure');
subplot(3,1,2);
plot(F0tdc, 'Displayname', 'Temperature');
xlabel('Temperature');
subplot(3,1,3);
plot(F0vibration, 'Displayname', 'Vibration');
xlabel('Vibration');
%% electrodes
figure;
for i = 1:1:19
    subplot(10,2,i);
    plot(F0Electrodes(i, :), 'Displayname', 'Electrodes');
    set(gca,'ytick',[],'yticklabel',[])
    set(gca,'xtick',[],'xticklabel',[])
    xlabel(i,'FontSize',8);
end

figure;
for i = 1:1:19
    plot(F0Electrodes(i, :), 'Displayname', 'Electrodes');
    hold on
end
hold off
legend('Electrode 1','Electrode 2','Electrode 3','Electrode 4','Electrode 5','Electrode 6','Electrode 7','Electrode 8','Electrode 9','Electrode 10','Electrode 11','Electrode 12','Electrode 13','Electrode 14','Electrode 15','Electrode 16','Electrode 17','Electrode 18','Electrode 19');
%title('Electrodes');
