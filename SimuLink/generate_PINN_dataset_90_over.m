clear
clc

warning('off','all');

spRoot = matlabshared.supportpkg.getSupportPackageRoot;

load(fullfile( ...
    spRoot,...
    'toolbox','autoblks','autodata','drivecycledata',...
    'cycleUS06.mat'));

PINN_CYCLE = cycleUS06;

fprintf('\n');
fprintf('=====================================================\n');
fprintf('PINN FULL DATASET GENERATION\n');
fprintf('=====================================================\n');

addpath(genpath( ...
    'C:\Users\Abimanyu\MATLAB\Projects\examples\VirtualVehicle2'));

rehash

load_system('Ebike_Thermal_DT_v1');

outputFolder = 'PINN_Dataset';

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

%% TRAIN / TEST SPLIT

cycles = { ...
'cycleFTP75'
'cycleUS06'
'cycleUDDS'
'cycleHUDDS'
'cycleWLTP1'
'cycleWLTP2'};

tempsC = [0 10 25 40 50];

socsPct = [10 20 40 60 80 100];

massScales = [0.8 1.0 1.2];

%% DRIVE CYCLE BLOCK

driveCycleBlock = ...
'Ebike_Thermal_DT_v1/Scenarios/Reference Generator/Drive Cycle/Drive Cycle Source';

set_param(driveCycleBlock,'cycleRepeat','on');

%% DATA DICTIONARY

dd = Simulink.data.dictionary.open('VirtualVehicleTemplate.sldd');
dData = getSection(dd,'Design Data');

origEnvTemp = getValue(getEntry(dData,'EnvAirTemp'));
origSOC = getValue(getEntry(dData,'PlntBattSocInit'));
origVehMass = getValue(getEntry(dData,'PlntVehMass'));

%% SUPPORT PACKAGE LOCATION

spRoot = matlabshared.supportpkg.getSupportPackageRoot;

%% TOTAL RUNS

nRuns = ...
length(cycles) * ...
length(tempsC) * ...
length(socsPct) * ...
length(massScales);

fprintf('Total simulations : %d\n',nRuns);
fprintf('\n');

runCounter = 0;
overallTimer = tic;

%% MAIN LOOP

for c = 1:length(cycles)

    cycleName = cycles{c};

    load(fullfile( ...
        spRoot,...
        'toolbox','autoblks','autodata','drivecycledata',...
        [cycleName '.mat']));

    eval(sprintf('PINN_CYCLE = %s;',cycleName));

    set_param(driveCycleBlock,'wsVar','PINN_CYCLE');
    set_param(driveCycleBlock,'cycleVar','Workspace variable');

    for t = 1:length(tempsC)

        tempC = tempsC(t);

        for s = 1:length(socsPct)

            socPct = socsPct(s);

            %% KNOWN BAD OPERATING POINT

            if tempC == 0 && socPct == 10
                fprintf('\nSKIPPING INVALID CASE T=0C SOC=10%%\n');
                continue
            end

            for m = 1:length(massScales)

                massScale = massScales(m);

                %% TRAIN / TEST LABEL

                if strcmp(cycleName,'cycleWLTP2')
                    splitName = 'TEST';
                else
                    splitName = 'TRAIN';
                end

                %% OUTPUT FILE NAME

                fileName = sprintf( ...
                    '%s_%s_T%d_SOC%d_M%.1f.csv', ...
                    splitName,...
                    cycleName,...
                    tempC,...
                    socPct,...
                    massScale);

                fullName = fullfile(outputFolder,fileName);

                %% RESUME SUPPORT

                if exist(fullName,'file')
                    fprintf('SKIP EXISTING : %s\n',fileName);
                    continue
                end

                runCounter = runCounter + 1;

                %% PERIODIC MODEL REFRESH

                if mod(runCounter,25)==0

                    fprintf('\n');
                    fprintf('Refreshing model...\n');

                    bdclose('all');
                    load_system('Ebike_Thermal_DT_v1');

                    driveCycleBlock = ...
                        'Ebike_Thermal_DT_v1/Scenarios/Reference Generator/Drive Cycle/Drive Cycle Source';

                    set_param(driveCycleBlock,'cycleRepeat','on');

                    set_param(driveCycleBlock,'wsVar','PINN_CYCLE');
                    set_param(driveCycleBlock,'cycleVar','Workspace variable');

                end

                elapsed = toc(overallTimer);

                if runCounter > 1
                    avgTime = elapsed/(runCounter-1);
                    etaMin = avgTime*(nRuns-runCounter)/60;
                else
                    etaMin = 0;
                end

                fprintf('\n');
                fprintf('[%d/%d] ',runCounter,nRuns);
                fprintf('%s ',cycleName);
                fprintf('| Temp=%dC ',tempC);
                fprintf('| SOC=%d%% ',socPct);
                fprintf('| Mass=%.1f ',massScale);
                fprintf('| ETA %.1f min\n',etaMin);

                try

                    %% PARAMETERS

                    setValue( ...
                        getEntry(dData,'EnvAirTemp'), ...
                        tempC + 273.15);

                    setValue( ...
                        getEntry(dData,'PlntBattSocInit'), ...
                        socPct/100);

                    setValue( ...
                        getEntry(dData,'PlntVehMass'), ...
                        origVehMass * massScale);

                    saveChanges(dd);

                    %% SIMULATION

                    out = sim( ...
                        'Ebike_Thermal_DT_v1', ...
                        'ReturnWorkspaceOutputs','on');

                    %% SIGNAL EXTRACTION

                    Batt = out.logsout{1}.Values;

                    Time_s = Batt.BattTemp.Time(:);
                    Current_A = Batt.BattCurr.Data(:);
                    Voltage_V = Batt.BattVolt.Data(:);
                    SOC_pct = Batt.BattSoc.Data(:);
                    BatteryPower_W = Batt.BattPwr.Data(:);
                    PowerLoss_W = Batt.BattPwrLoss.Data(:);
                    Temperature_C = Batt.BattTemp.Data(:)-273.15;

                    AmbientTemp_C = repmat(tempC,length(Time_s),1);
                    InitialSOC_pct = repmat(socPct,length(Time_s),1);
                    InitialBattTemp_C = repmat(tempC,length(Time_s),1);
                    MassScale = repmat(massScale,length(Time_s),1);
                    DriveCycle = repmat(string(cycleName),length(Time_s),1);

                    %% TABLE

                    T = table( ...
                        Time_s,...
                        Current_A,...
                        Voltage_V,...
                        SOC_pct,...
                        BatteryPower_W,...
                        PowerLoss_W,...
                        Temperature_C,...
                        AmbientTemp_C,...
                        InitialSOC_pct,...
                        InitialBattTemp_C,...
                        MassScale,...
                        DriveCycle);

                    %% SAVE

                    writetable(T,fullName);

                    fprintf('SAVED : %s\n',fileName);

                    drawnow;

                    clear out Batt T Time_s Current_A Voltage_V SOC_pct ...
                        BatteryPower_W PowerLoss_W Temperature_C ...
                        AmbientTemp_C InitialSOC_pct InitialBattTemp_C ...
                        MassScale DriveCycle
                catch ME

                    fprintf('\n');
                    fprintf('FAILED : %s\n',fileName);
                    fprintf('%s\n',ME.message);
                    fprintf('\n');
                
                    clear out Batt T Time_s Current_A Voltage_V SOC_pct ...
                        BatteryPower_W PowerLoss_W Temperature_C ...
                        AmbientTemp_C InitialSOC_pct InitialBattTemp_C ...
                        MassScale DriveCycle
                
                    bdclose('all');
                    pause(2)
                
                    load_system('Ebike_Thermal_DT_v1');
                
                    driveCycleBlock = ...
                    'Ebike_Thermal_DT_v1/Scenarios/Reference Generator/Drive Cycle/Drive Cycle Source';
                
                    set_param(driveCycleBlock,'cycleRepeat','on');
                
                    set_param(driveCycleBlock,'wsVar','PINN_CYCLE');
                    set_param(driveCycleBlock,'cycleVar','Workspace variable');
                
                    continue
                
                end

            end
        end
    end
end

%% RESTORE ORIGINAL VALUES

setValue(getEntry(dData,'EnvAirTemp'),origEnvTemp);
setValue(getEntry(dData,'PlntBattSocInit'),origSOC);
setValue(getEntry(dData,'PlntVehMass'),origVehMass);

saveChanges(dd);

%% FINAL SUMMARY

elapsedHours = toc(overallTimer)/3600;

fprintf('\n');
fprintf('=====================================================\n');
fprintf('DATASET GENERATION COMPLETE\n');
fprintf('=====================================================\n');
fprintf('Total simulations : %d\n',nRuns);
fprintf('Elapsed time      : %.2f hours\n',elapsedHours);
fprintf('Output folder     : %s\n',outputFolder);
fprintf('=====================================================\n');