# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_1 (link6_sensor_5, signal 79.4%)

## Which links/sensors carry the strongest valid signal?
Top links: link1 (21.1%, 2065 valid sensor-frames), link6 (15.0%, 1770 valid sensor-frames), link5 (8.6%, 2950 valid sensor-frames), link4 (6.8%, 1475 valid sensor-frames), link3 (0.3%, 1475 valid sensor-frames)
Top sensors: link1_sensor_5 (99.3%), link1_sensor_0 (48.5%), link5_back_sensor_5 (44.1%), link6_sensor_5 (31.9%), link6_sensor_4 (23.7%), link4_sensor_0 (21.4%), link5_front_sensor_0 (18.6%), link6_sensor_3 (14.2%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 11.4% over 816 valid frames; pregrasp: n/a over 0 valid frames; grasp_lift: 23.5% over 1552 valid frames; transit: n/a over 0 valid frames; place: n/a over 0 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
3 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_1 link1_sensor_5 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_0 link1_sensor_5 (activation_lt_0_20m_close_to_1), house_1 link5_back_sensor_5 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
