from __future__ import annotations

import argparse
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from libs.config.loader import load_yaml, require
from libs.obslog import configure as configure_logging

from .domain_health import DomainHealthConfig, DomainHealthTracker, make_stats_loader
from .service import OffererDerivation, OffererConfig, OffererService
from .selection.golden_discovery_ranker_v1_strategy import GoldenDiscoveryRankerV1Strategy
from .selection.example_strategy import ExampleStrategy
from .selection.read_only_strategy import ReadOnlyStrategy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--offerer-id", type=int, required=True)
    args = ap.parse_args()

    configure_logging(service="offerer", worker_id=args.offerer_id)
    raw = load_yaml(args.config)

    offerer = require(raw, "offerer")
    pg = require(raw, "postgres")

    id_start = int(offerer.get("id_start", 0))
    id_end = int(offerer.get("id_end", 0))
    offerer_id = int(args.offerer_id)

    if not (id_start <= offerer_id <= id_end):
        raise SystemExit(f"offerer-id {offerer_id} not in configured range [{id_start}, {id_end}]")

    engine = create_engine(
        str(require(pg, "dsn")),
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=2,
        max_overflow=1,
        pool_timeout=30,
        future=True,
        connect_args={
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 5,
            "keepalives_count": 5
        },
    )
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    strategy_name = os.environ.get("OFFERER_STRATEGY", "").lower()
    if strategy_name == "read_only":
        selector = ReadOnlyStrategy(Session=Session)
    elif strategy_name == "golden_discovery_ranker_v1":
        selector = GoldenDiscoveryRankerV1Strategy(Session=Session)
    else:
        selector = ExampleStrategy(Session=Session)

    deriv = OffererDerivation(
        queue_dir_template=str(offerer.get("queue_dir_template", "/data/ipc/url_queue/crawler_{id:02d}")),
        total_shards=int(offerer.get("total_shards", 256)),
        shards_per_offerer=int(offerer.get("shards_per_offerer", 16)),
    )

    cfg = OffererConfig(
        offerer_id=offerer_id,
        scan_interval_sec=int(offerer.get("scan_interval_sec", 300)),
        max_domain_files=int(offerer.get("max_domain_files", 32)),
        low_watermark_domains=int(offerer.get("low_watermark_domains", 16)),
        per_domain_url_cap=int(offerer.get("per_domain_url_cap", 100)),
        stats_dir=str(offerer.get("stats_dir", "/data/ipc/stats")),
        peek_multiplier=int(offerer.get("peek_multiplier", 3)),
        peek_hard_cap=int(offerer.get("peek_hard_cap", 1000)),
    )

    health_raw = offerer.get("domain_health") or {}
    health_enabled = os.environ.get("OFFERER_DOMAIN_HEALTH_ENABLED")
    health_cfg = DomainHealthConfig(
        enabled=(
            health_enabled.strip().lower() in {"1", "true", "yes", "on"}
            if health_enabled is not None
            else bool(health_raw.get("enabled", False))
        ),
        window_days=int(health_raw.get("window_days", 3)),
        refresh_sec=int(health_raw.get("refresh_sec", 900)),
        min_results=int(health_raw.get("min_results", 50)),
        min_ok_rate=float(health_raw.get("min_ok_rate", 0.2)),
        min_scheduled=int(health_raw.get("min_scheduled", 1000)),
        min_response_rate=float(health_raw.get("min_response_rate", 0.05)),
        probe_interval_sec=int(health_raw.get("probe_interval_sec", 21600)),
        pipeline_min_response_rate=float(health_raw.get("pipeline_min_response_rate", 0.5)),
    )
    health = (
        DomainHealthTracker(health_cfg, make_stats_loader(Session))
        if health_cfg.enabled else None
    )

    OffererService(cfg, deriv, selector, health=health).run_forever()


if __name__ == "__main__":
    main()
