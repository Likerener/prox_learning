# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_12 (link5_sensor_1, signal 100.0%), house_13 (link6_sensor_4, signal 100.0%), house_9 (link5_sensor_3, signal 100.0%), house_5 (link6_sensor_4, signal 100.0%), house_33 (link6_sensor_3, signal 79.8%), house_11 (link5_sensor_3, signal 78.0%), house_25 (link5_sensor_3, signal 75.5%), house_23 (link5_sensor_3, signal 67.2%), house_6 (link5_sensor_1, signal 65.7%), house_0 (link6_sensor_6, signal 63.8%), house_14 (link5_sensor_4, signal 37.9%), house_16 (link5_sensor_5, signal 33.1%)

## Which links/sensors carry the strongest valid signal?
Top links: link5 (18.8%, 23044 valid sensor-frames), link2 (14.9%, 27641 valid sensor-frames), link3 (13.4%, 31405 valid sensor-frames), link6 (12.9%, 31484 valid sensor-frames)
Top sensors: link5_sensor_3 (24.8%), link5_sensor_4 (23.4%), link2_sensor_0 (22.9%), link3_sensor_1 (20.3%), link3_sensor_5 (20.1%), link2_sensor_6 (19.4%), link5_sensor_1 (18.6%), link3_sensor_4 (18.5%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 9.4% over 1175 valid frames; pregrasp: 15.2% over 13096 valid frames; grasp_lift: 21.1% over 14728 valid frames; transit: 16.1% over 3604 valid frames; place: 7.7% over 4090 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
59 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_14 link6_sensor_0 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_14 link6_sensor_1 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_14 link6_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_6 link2_sensor_0 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_6 link2_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_0 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_3 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_5 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
