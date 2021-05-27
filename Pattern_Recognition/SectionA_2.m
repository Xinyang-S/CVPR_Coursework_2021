%% Section A2
Data = []; %comment this line after first trail, 60 trails in total
i = 50;

unit_pvt = [F0pdc(i); F0vibration(i); F0tdc(i)];
Data_pvt = [Data_pvt, unit_pvt]

unit_ele = [F0Electrodes(:, i)];
Data_ele = [Data_ele, unit_ele]