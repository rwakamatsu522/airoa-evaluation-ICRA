# REPRODUCTION_STEPS — Team 26 (TANABE × SideWide)

## 1. Overview

| Item | Value |
|---|---|
| Repository | `https://github.com/rwakamatsu522/airoa-evaluation-ICRA` |
| Branch | `feat/team26-lerobot-pi0` |
| Checkpoint S3 path | `s3://airoa-icra-team-26/team26-round4-checkpoint/` (Cloudflare R2) |

## 2. Steps

```bash
# 1. Clone (default branch is feat/team26-lerobot-pi0)
git clone https://github.com/rwakamatsu522/airoa-evaluation-ICRA.git
cd airoa-evaluation-ICRA

# 2. Download checkpoint (~14 GB) from Cloudflare R2
mkdir -p checkpoint
aws --endpoint-url https://eabeb2a5516ef53a191452e5714fc16b.r2.cloudflarestorage.com \
    s3 sync s3://airoa-icra-team-26/team26-round4-checkpoint/ ./checkpoint/

# 3. Env vars
export POLICY_CHECKPOINT_PATH=$(pwd)/checkpoint

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
├── adapter_config.json
├── adapter_model.safetensors
├── base_model/            # pi0 base + tokenizer (bundled, no HF_TOKEN required)
└── stats.json             # normalization
```

## 4. Contact

Team 26 (TANABE × SideWide) — submission 2026-04-15 (Round 4).
