clear
clc

model = 'Ebike_Thermal_DT_v1';

driveCycleBlock = ...
'Ebike_Thermal_DT_v1/Scenarios/Reference Generator/Drive Cycle/Drive Cycle Source';

spRoot = matlabshared.supportpkg.getSupportPackageRoot;

dd = Simulink.data.dictionary.open('VirtualVehicleTemplate.sldd');
dData = getSection(dd,'Design Data');

baseMass = getValue(getEntry(dData,'PlntVehMass'));

origAmbient  = getValue(getEntry(dData,'EnvAirTempC'));
origBattTemp = getValue(getEntry(dData,'BattTempInit'));
origBattTemp2 = getValue(getEntry(dData,'PlntBattTempInitDegC'));
origSOC      = getValue(getEntry(dData,'PlntBattSocInit'));

cycleFiles = { ...
    'cycleFTP75.mat'
    'cycleUS06.mat'
    'cycleUDDS.mat'
    'cycleHUDDS.mat'
    'cycleWLTP1.mat'
    'cycleWLTP2.mat'};

ambientTemps = [10 25 50];

initTemps = [30];

socs = [0.6];

massScales = [1.0];

totalRuns = ...
    length(cycleFiles) * ...
    length(ambientTemps) * ...
    length(initTemps) * ...
    length(socs) * ...
    length(massScales);

outputFolder = 'PINN_Ambient_Test';

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

load_system(model)

runID = 1;

tic

for c = 1:length(cycleFiles)

    S = load(fullfile( ...
        spRoot, ...
        'toolbox', ...
        'autoblks', ...
        'autodata', ...
        'drivecycledata', ...
        cycleFiles{c}));

    fn = fieldnames(S);

    cycleData = S.(fn{1});

    cycleName = erase(cycleFiles{c},'.mat');

    assignin('base',cycleName,cycleData);

    set_param(driveCycleBlock,...
        'wsVar',cycleName);

    set_param(driveCycleBlock,...
        'cycleVar','Workspace variable');

    for amb = ambientTemps

        setValue(getEntry(dData,'EnvAirTempC'),amb);

        for T0 = initTemps

            setValue(getEntry(dData,'BattTempInit'),T0);
            setValue(getEntry(dData,'PlntBattTempInitDegC'),T0);

            for soc = socs

                setValue(getEntry(dData,'PlntBattSocInit'),soc);

                for ms = massScales

                    setValue( ...
                        getEntry(dData,'PlntVehMass'), ...
                        baseMass * ms);

                    saveChanges(dd);

                    elapsed = toc;
                    avgTime = elapsed / max(runID-1,1);
                    remaining = avgTime * (totalRuns-runID);

                    fprintf( ...
                        '[%d/%d] %s | Ambient=%dC | ETA %.1f min\n', ...
                        runID, ...
                        totalRuns, ...
                        cycleName, ...
                        amb, ...
                        remaining/60);

                    out = sim(model,'ReturnWorkspaceOutputs','on');

                    Batt = out.logsout{1}.Values;

                    battTemp = Batt.BattTemp;
                    battCurr = Batt.BattCurr;
                    battVolt = Batt.BattVolt;
                    battSOC  = Batt.BattSoc;
                    battPwr  = Batt.BattPwr;
                    battLoss = Batt.BattPwrLoss;

                    Dataset = table( ...
                        battTemp.Time, ...
                        battCurr.Data, ...
                        battVolt.Data, ...
                        battSOC.Data, ...
                        battPwr.Data, ...
                        battLoss.Data, ...
                        battTemp.Data - 273.15, ...
                        repmat(amb,length(battTemp.Time),1), ...
                        'VariableNames',{ ...
                        'Time', ...
                        'Current', ...
                        'Voltage', ...
                        'SOC', ...
                        'BatteryPower', ...
                        'PowerLoss', ...
                        'Temperature_C', ...
                        'AmbientTemp_C'});

                    fname = sprintf( ...
                        '%s_A%d.csv', ...
                        cycleName, ...
                        amb);

                    writetable( ...
                        Dataset, ...
                        fullfile(outputFolder,fname));

                    Tmax = max(Dataset.Temperature_C);
                    Tmin = min(Dataset.Temperature_C);

                    fprintf( ...
                        'Saved %s | Temp Range %.3f -> %.3f C\n', ...
                        fname, ...
                        Tmin, ...
                        Tmax);

                    runID = runID + 1;

                end
            end
        end
    end
end

setValue(getEntry(dData,'EnvAirTempC'),origAmbient);
setValue(getEntry(dData,'BattTempInit'),origBattTemp);
setValue(getEntry(dData,'PlntBattTempInitDegC'),origBattTemp2);
setValue(getEntry(dData,'PlntBattSocInit'),origSOC);
setValue(getEntry(dData,'PlntVehMass'),baseMass);

saveChanges(dd);

disp('Ambient sensitivity study complete.')