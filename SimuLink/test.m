dd = Simulink.data.dictionary.open('VirtualVehicleTemplate.sldd');
dData = getSection(dd,'Design Data');

origMass = getValue(getEntry(dData,'PlntBattMass'));

tests = [0.5 1.0 2.0];

for s = tests

    setValue(getEntry(dData,'PlntBattMass'),origMass*s);
    saveChanges(dd);

    out = sim('Ebike_Thermal_DT_v1','ReturnWorkspaceOutputs','on');

    Batt = out.logsout{1}.Values;

    tempC = Batt.BattTemp.Data - 273.15;

    fprintf('\n');
    fprintf('Battery Mass Scale %.1f\n',s);
    fprintf('Start %.3f C\n',tempC(1));
    fprintf('Max   %.3f C\n',max(tempC));
    fprintf('Rise  %.3f C\n',max(tempC)-tempC(1));

end

setValue(getEntry(dData,'PlntBattMass'),origMass);
saveChanges(dd);