# VisionGuard

[![CI](https://github.com/nabihanoor19/23L-0967_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/nabihanoor19/23L-0967_A2_Ops/actions/workflows/ci.yml)

VisionGuard tells whether an uploaded image is AI-generated or real. It uses a MobileNetV2 transfer-learning model, a Flask web app and a PostgreSQL history, packaged with Docker Compose. Its verdicts are estimates, not proof.

## Run with Docker
1. Copy `.env.example` to `.env` and fill in the values.
2. Put `champion.keras` and `champion_meta.json` in `models/` (`python -m scripts.download_champion`).
3. Run `docker compose up -d --build` and open http://localhost:5000.