batteryThermalPath = ...
    'Ebike_Thermal_DT_v1/Vehicle/ConfiguredSimulinkPlantModel/Battery/BatteryMapped/Thermal and Accessory Power System';

disp('=================================================')
disp('THERMAL SUBSYSTEM PARAMETERS')
disp('=================================================')

get_param(batteryThermalPath,'ObjectParameters')

disp('=================================================')
disp('TEMP INTEGRATOR BLOCK')
disp('=================================================')

tempInt = [batteryThermalPath '/TempIntegrator'];

get_param(tempInt,'BlockType')

try
    get_param(tempInt,'InitialCondition')
catch
    disp('No InitialCondition parameter')
end

disp('=================================================')
disp('BATTTEMPOUT BLOCK')
disp('=================================================')

battOut = [batteryThermalPath '/BattTempOut'];

try
    get_param(battOut,'BlockType')
catch
end

disp('=================================================')
disp('MASKS INSIDE THERMAL SUBSYSTEM')
disp('=================================================')

find_system( ...
    batteryThermalPath, ...
    'LookUnderMasks','all', ...
    'FollowLinks','on', ...
    'Mask','on')