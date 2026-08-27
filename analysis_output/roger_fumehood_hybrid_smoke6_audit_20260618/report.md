# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_0 (link6_sensor_5, signal 86.5%), house_1 (link6_sensor_5, signal 80.3%)

## Which links/sensors carry the strongest valid signal?
Top links: link1 (20.6%, 7476 valid sensor-frames), link6 (16.8%, 6408 valid sensor-frames), link5 (8.0%, 10680 valid sensor-frames), link4 (1.1%, 5340 valid sensor-frames), link3 (0.5%, 5340 valid sensor-frames)
Top sensors: link1_sensor_5 (99.8%), link5_back_sensor_5 (58.4%), link1_sensor_0 (44.3%), link6_sensor_5 (35.2%), link6_sensor_2 (20.2%), link6_sensor_0 (15.5%), link6_sensor_4 (15.3%), link6_sensor_3 (14.4%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 9.3% over 1792 valid frames; pregrasp: n/a over 0 valid frames; grasp_lift: 22.9% over 5728 valid frames; transit: n/a over 0 valid frames; place: n/a over 0 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
4 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_1 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation;activation_spread_across_many_phases), house_0 link1_sensor_5 (activation_lt_0_20m_close_to_1;extremely_low_frame_to_frame_variation;activation_spread_across_many_phases), house_0 link5_back_sensor_5 (activation_spread_across_many_phases), house_1 link5_back_sensor_5 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
