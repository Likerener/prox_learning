# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_120 (link5_sensor_3, signal 98.4%), house_280 (link5_sensor_3, signal 48.3%)

## Which links/sensors carry the strongest valid signal?
Top links: link3 (10.2%, 4240 valid sensor-frames), link2 (6.4%, 3710 valid sensor-frames), link5 (2.8%, 3180 valid sensor-frames), link6 (1.4%, 4240 valid sensor-frames)
Top sensors: link2_sensor_0 (27.0%), link3_sensor_6 (18.1%), link2_sensor_2 (17.7%), link5_sensor_3 (16.6%), link3_sensor_2 (14.2%), link3_sensor_4 (13.8%), link3_sensor_5 (13.0%), link3_sensor_0 (12.8%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 0.0% over 154 valid frames; pregrasp: 0.1% over 1022 valid frames; grasp_lift: 8.8% over 1666 valid frames; transit: 0.0% over 490 valid frames; place: 0.0% over 840 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
1 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_120 link2_sensor_0 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Do not use the prior activation signal as evidence: link5/link6 valid activation did not survive filtering.
