# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_16 (link5_sensor_3, signal 100.0%), house_35 (link6_sensor_4, signal 100.0%), house_20 (link6_sensor_1, signal 100.0%), house_29 (link6_sensor_0, signal 100.0%), house_12 (link6_sensor_3, signal 98.0%), house_33 (link6_sensor_5, signal 76.9%), house_23 (link5_sensor_1, signal 71.6%), house_1 (link5_sensor_3, signal 42.6%), house_21 (link6_sensor_6, signal 15.5%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (18.3%, 31048 valid sensor-frames), link5 (2.3%, 23286 valid sensor-frames), link2 (2.0%, 27167 valid sensor-frames), link3 (0.0%, 31034 valid sensor-frames)
Top sensors: link6_sensor_0 (25.1%), link6_sensor_1 (24.8%), link6_sensor_6 (24.7%), link6_sensor_5 (23.7%), link6_sensor_2 (23.4%), link6_sensor_4 (15.0%), link5_sensor_3 (7.3%), link6_sensor_3 (7.0%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 10.8% over 1176 valid frames; pregrasp: 9.5% over 7196 valid frames; grasp_lift: 21.3% over 13762 valid frames; transit: 15.2% over 3906 valid frames; place: 9.8% over 4480 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
31 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_1 link6_sensor_0 (activation_spread_across_many_phases), house_1 link6_sensor_6 (activation_spread_across_many_phases), house_1 link6_sensor_1 (activation_spread_across_many_phases), house_14 link6_sensor_0 (activation_spread_across_many_phases), house_14 link6_sensor_1 (activation_spread_across_many_phases), house_14 link6_sensor_6 (activation_spread_across_many_phases), house_14 link6_sensor_2 (activation_spread_across_many_phases), house_14 link6_sensor_5 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
