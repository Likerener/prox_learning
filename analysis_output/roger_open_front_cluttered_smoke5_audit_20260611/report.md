# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_9 (link5_sensor_3, signal 100.0%), house_11 (link5_sensor_3, signal 71.7%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (3.6%, 11056 valid sensor-frames), link3 (2.6%, 11052 valid sensor-frames), link5 (2.6%, 8292 valid sensor-frames), link2 (2.0%, 9674 valid sensor-frames)
Top sensors: link3_sensor_2 (18.7%), link2_sensor_0 (12.4%), link5_sensor_3 (11.4%), link6_sensor_0 (5.5%), link6_sensor_6 (4.6%), link6_sensor_3 (4.3%), link6_sensor_1 (4.1%), link6_sensor_5 (3.8%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 7.4% over 406 valid frames; pregrasp: 0.5% over 4242 valid frames; grasp_lift: 7.8% over 4018 valid frames; transit: 2.5% over 994 valid frames; place: 6.1% over 1596 valid frames.
Activation is not concentrated only in pregrasp/grasp_lift; inspect the phase table for spread.

## Which rows should be excluded?
1 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_7 link3_sensor_2 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Compared with the old audit, 0 rows had zero/negative old min depths versus 0 after filtering. Max absolute delta in <0.20m activation was 34.1%; max delta in <0.05m near-saturation was 0.1%.
Material-change flag: yes. Treat cleaned numbers as authoritative.

## Decision
Do not use the prior activation signal as evidence: link5/link6 valid activation did not survive filtering.
