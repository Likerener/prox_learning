# Proximity Activation Audit (cleaned)

Valid depth rule: `(0.00, 4.00] m`; activation rates use valid frames only.

## Which houses have useful valid proximity activation?
house_14 (link5_sensor_3, signal 100.0%), house_15 (link6_sensor_1, signal 100.0%), house_13 (link6_sensor_1, signal 91.7%), house_23 (link6_sensor_0, signal 67.3%), house_17 (link5_sensor_3, signal 26.4%), house_24 (link5_sensor_3, signal 23.4%)

## Which links/sensors carry the strongest valid signal?
Top links: link6 (20.6%, 18995 valid sensor-frames), link5 (19.1%, 14215 valid sensor-frames), link2 (12.9%, 16651 valid sensor-frames), link3 (9.5%, 19031 valid sensor-frames)
Top sensors: link5_sensor_3 (29.5%), link6_sensor_4 (29.0%), link6_sensor_3 (28.6%), link6_sensor_5 (26.2%), link5_sensor_4 (23.7%), link6_sensor_2 (22.0%), link2_sensor_0 (21.6%), link6_sensor_7 (21.1%)

## Does activation concentrate in pregrasp and grasp_lift?
For link5/link6, valid-frame activation <0.20m by phase is: approach: 22.4% over 732 valid frames; pregrasp: 13.6% over 8148 valid frames; grasp_lift: 27.7% over 8580 valid frames; transit: 24.1% over 2057 valid frames; place: 20.1% over 2818 valid frames.
Activation is not concentrated only in pregrasp/grasp_lift; inspect the phase table for spread.

## Which rows should be excluded?
39 house/sensor rows were flagged. The CSV gives exact reasons; top examples: house_13 link2_sensor_0 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_3 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_4 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_5 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link2_sensor_6 (activation_lt_0_20m_close_to_1;activation_spread_across_many_phases), house_13 link3_sensor_5 (activation_spread_across_many_phases), house_13 link3_sensor_1 (activation_spread_across_many_phases), house_13 link3_sensor_7 (activation_spread_across_many_phases)

## Did filtering materially change the old audit?
Compared with the old audit, 0 rows had zero/negative old min depths versus 0 after filtering. Max absolute delta in <0.20m activation was 25.3%; max delta in <0.05m near-saturation was 6.2%.
Material-change flag: yes. Treat cleaned numbers as authoritative.

## Decision
Keep the environment: link5/link6 retain meaningful valid activation in pregrasp or grasp_lift.
