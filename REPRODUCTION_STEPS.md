# REPRODUCTION_STEPS — Team 26 Round 5 (TANABE × SideWide)

## 1. Overview

| Item | Value |
|---|---|
| Repository | `https://github.com/rwakamatsu522/airoa-evaluation-ICRA` |
| Branch | `submission/round5` |
| Checkpoint S3 path | `s3://airoa-icra-team-26/team26-round5-checkpoint/` (Cloudflare R2) |
| Config name | `pi05_hsr_task6911` |
| Framework | OpenPI (pi05 / JAX) |

## 2. Steps

```bash
# 1. Clone & checkout
git clone https://github.com/rwakamatsu522/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA
git checkout submission/round5

# 2. Download checkpoint (~12 GB) from Cloudflare R2
mkdir -p checkpoint
aws --endpoint-url https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com \
    s3 sync s3://airoa-icra-team-26/team26-round5-checkpoint/ ./checkpoint/

# 3. Env vars
export POLICY_CHECKPOINT_DIR=$(pwd)/checkpoint
export POLICY_CONFIG_NAME=pi05_hsr_task6911

# 4. Start (first build ~10–15 min, model load ~60 s)
./RUN-DOCKER-CONTAINER.sh up

# 5. Verify server ready
until curl -s http://localhost:8000/healthz | grep -q OK; do sleep 5; done
echo "READY"

# 6. Smoke test
./RUN-DOCKER-CONTAINER.sh shell
# inside container:
roslaunch hsr_policy_client hsr_policy_client.launch
# expect: "Action executed."

# 7. Stop
./RUN-DOCKER-CONTAINER.sh down
```

## 3. Checkpoint layout

```
checkpoint/
├── _CHECKPOINT_METADATA       # orbax checkpoint metadata
├── assets/
│   └── task6911/
│       └── norm_stats.json    # normalization statistics
└── params/                    # model weights (orbax format, ~12 GB)
```
