# ToggleSwitch objective check

Primary metrics were fixed before fitting: mutual k=15 graph, anchor seed 714,
global distance rank and 15-neighbor overlap. Sensitivity metrics, not used
for selection, are k=20, k=30, anchor seed 716, and direct feature-distance
rank. Returned coordinates are the lowest training loss of that objective.
Losses are not compared across objectives. The corrected self-inclusive
objective is the baseline. The historical probability floor was not run.

Input SHA256
`25769c7044c1411a043ce00c41177aaf1e2091a950106caaa5976d7649e94fc1`.
Two hundred cells, k=15, sigma=1, gamma=2, 200 updates, seeds 0–4. Each run
stores 200 gradient-norm values and radius quantiles. This is not full
Planaria. Values are in [toggle-evaluation.json](toggle-evaluation.json).

| Anchor | Objective | Best loss | k=15 global | k=20 global | k=30 global | Feature rank |
|---|---|---:|---:|---:|---:|---:|
| 714 | symmetric KL | 354.61–354.65 | 0.8123 | 0.8100 | 0.8086 | 0.7149–0.7150 |
| 714 | conditional KL | 332.79–332.83 | 0.8085–0.8086 | 0.8062–0.8064 | 0.8048–0.8050 | 0.7111–0.7113 |
| 716 | symmetric KL | 354.61–354.65 | 0.8189–0.8189 | 0.8167–0.8168 | 0.8153–0.8154 | 0.7282–0.7283 |
| 716 | conditional KL | 332.79–332.83 | 0.8152–0.8153 | 0.8130–0.8131 | 0.8117–0.8118 | 0.7248–0.7249 |

Symmetric KL keeps a slightly higher distance rank at every declared graph
size and both anchor seeds. Changing k or the anchor seed does not reverse
that order. The conditional loss number is smaller because the objective
changed, not because a shared score improved. No default change follows.
