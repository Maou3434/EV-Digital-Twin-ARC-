clc

temps = [0 10 25 40 50];
socs  = [10 20 40 60 80 100];

for t = temps
    for s = socs

        fprintf('\nTesting T=%dC SOC=%d%%\n',t,s);

        try

            dd = Simulink.data.dictionary.open('VirtualVehicleTemplate.sldd');
            dData = getSection(dd,'Design Data');

            setValue(getEntry(dData,'EnvAirTemp'),t+273.15);
            setValue(getEntry(dData,'PlntBattSocInit'),s/100);

            saveChanges(dd);

            out = sim('Ebike_Thermal_DT_v1',...
                'ReturnWorkspaceOutputs','on');

            fprintf('PASS\n');

        catch ME

            fprintf('FAIL\n');
            fprintf('%s\n',ME.message);

        end
    end
end