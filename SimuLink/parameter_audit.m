clear
clc

model = 'Ebike_Thermal_DT_v1';

dd = Simulink.data.dictionary.open('VirtualVehicleTemplate.sldd');
dData = getSection(dd,'Design Data');

origAmbient  = getValue(getEntry(dData,'EnvAirTempC'));
origBattTemp = getValue(getEntry(dData,'BattTempInit'));
origBattTemp2 = getValue(getEntry(dData,'PlntBattTempInitDegC'));
origSOC      = getValue(getEntry(dData,'PlntBattSocInit'));
origMass     = getValue(getEntry(dData,'PlntVehMass'));

tests = {
    'Baseline',      25, 30, 0.6, origMass;
    'Ambient_50C',   50, 30, 0.6, origMass;
    'InitTemp_50C',  25, 50, 0.6, origMass;
    'SOC_100pct',    25, 30, 1.0, origMass;
    'Mass_120pct',   25, 30, 0.6, origMass*1.2;
    };

fprintf('\n');
fprintf('============================================================\n');
fprintf('THERMAL SENSITIVITY AUDIT\n');
fprintf('============================================================\n');

for k = 1:size(tests,1)

    name = tests{k,1};
    amb  = tests{k,2};
    T0   = tests{k,3};
    soc  = tests{k,4};
    mass = tests{k,5};

    setValue(getEntry(dData,'EnvAirTempC'),amb);
    setValue(getEntry(dData,'BattTempInit'),T0);
    setValue(getEntry(dData,'PlntBattTempInitDegC'),T0);
    setValue(getEntry(dData,'PlntBattSocInit'),soc);
    setValue(getEntry(dData,'PlntVehMass'),mass);

    saveChanges(dd);

    fprintf('\nRunning %s...\n',name);

    out = sim(model,'ReturnWorkspaceOutputs','on');

    Batt = out.logsout{1}.Values;

    tempC = Batt.BattTemp.Data - 273.15;

    fprintf('Start Temp : %.3f C\n',tempC(1));
    fprintf('Min Temp   : %.3f C\n',min(tempC));
    fprintf('Max Temp   : %.3f C\n',max(tempC));
    fprintf('Temp Rise  : %.3f C\n',max(tempC)-min(tempC));

    fprintf('Final SOC  : %.3f\n',Batt.BattSoc.Data(end));

    fprintf('Peak Current : %.3f A\n',max(abs(Batt.BattCurr.Data)));

    fprintf('Peak Power   : %.3f W\n',max(abs(Batt.BattPwr.Data)));

end

setValue(getEntry(dData,'EnvAirTempC'),origAmbient);
setValue(getEntry(dData,'BattTempInit'),origBattTemp);
setValue(getEntry(dData,'PlntBattTempInitDegC'),origBattTemp2);
setValue(getEntry(dData,'PlntBattSocInit'),origSOC);
setValue(getEntry(dData,'PlntVehMass'),origMass);

saveChanges(dd);

fprintf('\n');
fprintf('============================================================\n');
fprintf('AUDIT COMPLETE\n');
fprintf('============================================================\n');