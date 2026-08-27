# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_8 (link6_sensor_0, signal 100.0%), house_1 (link5_sensor_3, signal 74.1%)

## Which links/sensors carry the strongest valid signal?
Top links: link2 (6.4%, 9212 valid sensor-frames), link6 (3.6%, 10528 valid sensor-frames), link5 (1.3%, 7896 valid sensor-frames), link3 (0.5%, 10528 valid sensor-frames)
Top sensors: link2_sensor_6 (22.3%), link2_sensor_3 (12.5%), link2_sensor_0 (10.3%), link6_sensor_5 (6.2%), link6_sensor_1 (4.7%), link6_sensor_6 (4.7%), link6_sensor_2 (4.6%), link5_sensor_3 (4.4%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 1.6% over 364 valid frames; pregrasp: 1.4% over 2688 valid frames; grasp_lift: 9.5% over 3780 valid frames; transit: 2.8% over 1596 valid frames; place: 0.0% over 1876 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
2 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_1 link2_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_1 link2_sensor_3 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Old audit comparison unavailable: no_old_audit_dir_supplied.

## Decision
Do not use the prior activation signal as evidence: link5/link6 valid activation did not survive filtering.
