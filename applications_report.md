# Applications Report

Three economic applications of the LLM-scored Thai energy sentiment index.

## Application 1 — Granger Causality

Test whether sentiment Granger-causes fuel-price dynamics. Best-lag F-test reported.

| target                | predictor                       |   best_lag |   F_stat |   p_value | significant_5pct   |
|:----------------------|:--------------------------------|-----------:|---------:|----------:|:-------------------|
| Diesel volatility     | LLM sentiment (combined)        |          4 |   1.0810 |    0.3714 | False              |
| Diesel volatility     | LLM sentiment (English)         |          1 |   2.0165 |    0.1590 | False              |
| Diesel volatility     | LLM sentiment (Thai)            |          1 |   3.5948 |    0.0612 | False              |
| Diesel volatility     | Tetlock negativity (combined)   |          4 |   0.9264 |    0.4528 | False              |
| Diesel volatility     | GPR Global (Caldara-Iacoviello) |          4 |   1.7868 |    0.1394 | False              |
| Diesel volatility     | EPU US (first difference)       |          1 |   8.2110 |    0.0052 | True               |
| Gasohol 95 volatility | LLM sentiment (combined)        |          1 |   2.9815 |    0.0876 | False              |
| Gasohol 95 volatility | LLM sentiment (English)         |          2 |   4.1514 |    0.0189 | True               |
| Gasohol 95 volatility | LLM sentiment (Thai)            |          1 |   0.2740 |    0.6020 | False              |
| Gasohol 95 volatility | Tetlock negativity (combined)   |          1 |   4.1997 |    0.0433 | True               |
| Gasohol 95 volatility | GPR Global (Caldara-Iacoviello) |          2 |   1.1335 |    0.3266 | False              |
| Gasohol 95 volatility | EPU US (first difference)       |          2 |   3.6841 |    0.0291 | True               |
| Diesel log-return     | LLM sentiment (combined)        |          2 |   4.8347 |    0.0102 | True               |
| Diesel log-return     | LLM sentiment (English)         |          2 |   4.9497 |    0.0092 | True               |
| Diesel log-return     | LLM sentiment (Thai)            |          1 |  11.5394 |    0.0010 | True               |
| Diesel log-return     | Tetlock negativity (combined)   |          2 |   4.6574 |    0.0120 | True               |
| Diesel log-return     | GPR Global (Caldara-Iacoviello) |          2 |   0.5544 |    0.5764 | False              |
| Diesel log-return     | EPU US (first difference)       |          1 |   1.0015 |    0.3196 | False              |
| Gasohol 95 log-return | LLM sentiment (combined)        |          2 |   2.4700 |    0.0904 | False              |
| Gasohol 95 log-return | LLM sentiment (English)         |          2 |   5.1454 |    0.0077 | True               |
| Gasohol 95 log-return | LLM sentiment (Thai)            |          1 |  14.9974 |    0.0002 | True               |
| Gasohol 95 log-return | Tetlock negativity (combined)   |          2 |   1.9235 |    0.1522 | False              |
| Gasohol 95 log-return | GPR Global (Caldara-Iacoviello) |          4 |   0.8694 |    0.4860 | False              |
| Gasohol 95 log-return | EPU US (first difference)       |          3 |   3.9624 |    0.0107 | True               |


## Application 2 — Event Window: 2022 Russia-Ukraine Invasion

Sentiment and price path around the February 2022 invasion. Event month is 2022-02 (t_0).

| relative_month   | year_month   |   mean_sentiment_combined |   gasohol95_retail_bkk |   diesel_retail_bkk |   gpr_global |
|:-----------------|:-------------|--------------------------:|-----------------------:|--------------------:|-------------:|
| t-2              | 2021-12      |                    -0.038 |                 30.560 |              28.160 |      105.345 |
| t-1              | 2022-01      |                    -0.372 |                 32.580 |              29.580 |      138.675 |
| t_0              | 2022-02      |                    -0.332 |                 35.360 |              29.330 |      216.159 |
| t+1              | 2022-03      |                    -0.414 |                 39.100 |              29.900 |      318.955 |
| t+2              | 2022-04      |                    -0.257 |                 38.820 |              29.940 |      191.143 |
| t+3              | 2022-05      |                    -0.209 |                 42.100 |              31.970 |      142.258 |
| t+4              | 2022-06      |                    -0.263 |                 44.780 |              34.310 |      130.707 |
| t+5              | 2022-07      |                    -0.141 |                 40.050 |              34.940 |      117.177 |


## Application 3 — Encompassing Regression

OLS regression of 1-month-ahead fuel returns on sentiment, GPR, EPU. Tests whether LLM sentiment carries incremental predictive content.

| specification                             | dependent        |   n_obs |     R2 |   R2_adj |   F_pvalue |   beta_diesel_logret_lag1 |   p_diesel_logret_lag1 |   beta_sent_lag1 |   p_sent_lag1 |   beta_sent_tha_lag1 |   p_sent_tha_lag1 |   beta_gpr_lag1 |   p_gpr_lag1 |   beta_d_epu_us_lag1 |   p_d_epu_us_lag1 |   beta_gasohol95_logret_lag1 |   p_gasohol95_logret_lag1 |
|:------------------------------------------|:-----------------|--------:|-------:|---------:|-----------:|--------------------------:|-----------------------:|-----------------:|--------------:|---------------------:|------------------:|----------------:|-------------:|---------------------:|------------------:|-----------------------------:|--------------------------:|
| Diesel: AR(1) baseline                    | diesel_logret    |      94 | 0.1726 |   0.1636 |     0.0000 |                    0.4167 |                 0.0000 |         nan      |      nan      |             nan      |          nan      |        nan      |     nan      |             nan      |          nan      |                     nan      |                  nan      |
| Diesel: sentiment_lag1 only               | diesel_logret    |      95 | 0.0010 |  -0.0097 |     0.7576 |                  nan      |               nan      |          -0.0132 |        0.7576 |             nan      |          nan      |        nan      |     nan      |             nan      |          nan      |                     nan      |                  nan      |
| Diesel: sentiment_tha_lag1 only           | diesel_logret    |      93 | 0.2014 |   0.1926 |     0.0000 |                  nan      |               nan      |         nan      |      nan      |              -0.1063 |            0.0000 |        nan      |     nan      |             nan      |          nan      |                     nan      |                  nan      |
| Diesel: GPR_lag1 only                     | diesel_logret    |      95 | 0.0050 |  -0.0057 |     0.4979 |                  nan      |               nan      |         nan      |      nan      |             nan      |          nan      |          0.0001 |       0.4979 |             nan      |          nan      |                     nan      |                  nan      |
| Diesel: AR(1) + sentiment_lag1            | diesel_logret    |      94 | 0.1727 |   0.1545 |     0.0002 |                    0.4162 |                 0.0000 |          -0.0046 |        0.9086 |             nan      |          nan      |        nan      |     nan      |             nan      |          nan      |                     nan      |                  nan      |
| Diesel: AR(1) + sentiment_tha_lag1        | diesel_logret    |      92 | 0.2675 |   0.2510 |     0.0000 |                    0.2811 |                 0.0058 |         nan      |      nan      |              -0.0797 |            0.0010 |        nan      |     nan      |             nan      |          nan      |                     nan      |                  nan      |
| Diesel: AR(1) + sentiment + GPR + EPU     | diesel_logret    |      94 | 0.1852 |   0.1486 |     0.0010 |                    0.3764 |                 0.0004 |          -0.0015 |        0.9716 |             nan      |          nan      |          0.0001 |       0.5611 |              -0.0001 |            0.2984 |                     nan      |                  nan      |
| Gasohol 95: AR(1) baseline                | gasohol95_logret |      94 | 0.1139 |   0.1043 |     0.0009 |                  nan      |               nan      |         nan      |      nan      |             nan      |          nan      |        nan      |     nan      |             nan      |          nan      |                       0.3374 |                    0.0009 |
| Gasohol 95: sentiment_lag1 only           | gasohol95_logret |      95 | 0.0008 |  -0.0099 |     0.7795 |                  nan      |               nan      |          -0.0158 |        0.7795 |             nan      |          nan      |        nan      |     nan      |             nan      |          nan      |                     nan      |                  nan      |
| Gasohol 95: sentiment_tha_lag1 only       | gasohol95_logret |      93 | 0.2064 |   0.1977 |     0.0000 |                  nan      |               nan      |         nan      |      nan      |              -0.1422 |            0.0000 |        nan      |     nan      |             nan      |          nan      |                     nan      |                  nan      |
| Gasohol 95: AR(1) + sentiment_lag1        | gasohol95_logret |      94 | 0.1141 |   0.0946 |     0.0040 |                  nan      |               nan      |           0.0069 |        0.8995 |             nan      |          nan      |        nan      |     nan      |             nan      |          nan      |                       0.3391 |                    0.0010 |
| Gasohol 95: AR(1) + sentiment_tha_lag1    | gasohol95_logret |      92 | 0.2251 |   0.2077 |     0.0000 |                  nan      |               nan      |         nan      |      nan      |              -0.1189 |            0.0006 |        nan      |     nan      |             nan      |          nan      |                       0.1547 |                    0.1549 |
| Gasohol 95: AR(1) + sentiment + GPR + EPU | gasohol95_logret |      94 | 0.1571 |   0.1193 |     0.0040 |                  nan      |               nan      |          -0.0043 |        0.9402 |             nan      |          nan      |          0.0000 |       0.7874 |              -0.0002 |            0.0360 |                       0.2734 |                    0.0094 |