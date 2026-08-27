# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_1 (link5_sensor_3, signal 74.1%), house_8 (link6_sensor_0, signal 64.1%)

## Which links/sensors carry the strongest valid signal?
Top links: link2 (9.2%, 27069 valid sensor-frames), link6 (3.0%, 30936 valid sensor-frames), link5 (2.0%, 23202 valid sensor-frames), link3 (0.8%, 30936 valid sensor-frames)
Top sensors: link2_sensor_6 (34.9%), link2_sensor_3 (16.9%), link2_sensor_0 (12.2%), link5_sensor_3 (6.9%), link3_sensor_7 (5.9%), link6_sensor_5 (5.7%), link5_sensor_4 (4.2%), link6_sensor_0 (3.8%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 1.6% over 1092 valid frames; pregrasp: 1.9% over 7000 valid frames; grasp_lift: 8.4% over 11802 valid frames; transit: 2.2% over 4662 valid frames; place: 0.2% over 5222 valid frames.
By weighted valid-frame rate, pregrasp/grasp_lift exceed approach/transit/place.

## Which rows should be excluded?
3 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_1 link2_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_1 link2_sensor_3 (activation_spread_across_many_phases), house_8 link2_sensor_6 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Compared with the old audit, 0 rows had zero/negative old min depths versus 0 after filtering. Max absolute delta in <0.20m activation was 12.6%; max delta in <0.05m near-saturation was 0.1%.
Material-change flag: yes. Treat cleaned numbers as authoritative.

## Decision
Do not use the prior activation signal as evidence: link5/link6 valid activation did not survive filtering.
