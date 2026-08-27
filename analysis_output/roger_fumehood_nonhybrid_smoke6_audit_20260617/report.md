# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_1 (link6_sensor_7, signal 41.3%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (20.4%, 12552 valid sensor-frames), link5 (15.5%, 9414 valid sensor-frames), link2 (10.2%, 10983 valid sensor-frames), link3 (0.8%, 12552 valid sensor-frames)
Top sensors: link2_sensor_0 (66.0%), link5_sensor_3 (60.9%), link6_sensor_3 (35.6%), link6_sensor_7 (30.1%), link6_sensor_5 (22.3%), link6_sensor_0 (21.7%), link6_sensor_6 (17.6%), link6_sensor_4 (16.6%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 13.1% over 2072 valid frames; pregrasp: n/a over 0 valid frames; grasp_lift: 28.3% over 10066 valid frames; transit: n/a over 0 valid frames; place: n/a over 0 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
3 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_1 link2_sensor_0 (activation_spread_across_many_phases), house_1 link5_sensor_3 (activation_spread_across_many_phases), house_1 link6_sensor_3 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
