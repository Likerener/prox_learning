# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_14 (link5_sensor_0, signal 100.0%), house_16 (link6_sensor_5, signal 98.2%), house_3 (link6_sensor_4, signal 48.4%)

## Which links/sensors carry the strongest valid signal?
Top links: link5 (12.3%, 23081 valid sensor-frames), link2 (7.5%, 26950 valid sensor-frames), link6 (7.0%, 30800 valid sensor-frames), link3 (5.0%, 30800 valid sensor-frames)
Top sensors: link5_sensor_3 (17.2%), link6_sensor_4 (14.5%), link2_sensor_6 (13.7%), link5_sensor_4 (13.1%), link6_sensor_3 (13.1%), link5_sensor_5 (11.7%), link5_sensor_1 (11.2%), link6_sensor_7 (11.2%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 13.2% over 1064 valid frames; pregrasp: 7.1% over 6468 valid frames; grasp_lift: 8.7% over 11732 valid frames; transit: 6.2% over 4758 valid frames; place: 12.5% over 5510 valid frames.
Activation is not concentrated only in pregrasp/grasp_lift; inspect the phase table for spread.

## Which rows should be excluded?
19 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_16 link2_sensor_0 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_1 link5_sensor_3 (activation_spread_across_many_phases), house_16 link2_sensor_5 (activation_spread_across_many_phases), house_1 link6_sensor_3 (activation_spread_across_many_phases), house_1 link6_sensor_4 (activation_spread_across_many_phases), house_1 link5_sensor_4 (activation_spread_across_many_phases), house_16 link2_sensor_2 (activation_spread_across_many_phases), house_1 link6_sensor_7 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Do not use the prior activation signal as evidence: link5/link6 valid activation did not survive filtering.
