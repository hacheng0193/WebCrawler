"""Queue intake must not depend on owned requests finishing or the engine idling."""
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import scrapy
from scrapy.utils.test import get_crawler

# The spider imports `crawler.*` relative to the scrapy project root, as in the container.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crawler.spiders.spider import HtmlSpider  # noqa: E402


@pytest.fixture
def spider(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    queue_dir = tmp_path / "url_queue" / "crawler_00"
    queue_dir.mkdir(parents=True)
    crawler = get_crawler(HtmlSpider, {
        "URL_QUEUE_TEMPLATE": str(tmp_path / "url_queue" / "crawler_{id:02d}"),
        "MANUAL_BULK_CRAWL_QUEUE_DIR": str(tmp_path / "manual"),
    })
    spider = HtmlSpider.from_crawler(crawler, crawler_id=0)
    crawler.engine = Mock()
    crawler.engine.downloader.slots = {}
    spider.crawler = crawler

    def enqueue(domain_id, urls):
        (queue_dir / f"domain_{domain_id}.json").write_text(json.dumps({"urls": urls}))

    return spider, enqueue


def scheduled(spider):
    return [c.args[0] for c in spider.crawler.engine.crawl.call_args_list]


def test_periodic_top_up_reads_queue_while_engine_is_busy_with_other_work(spider):
    # Owned work drained while the queue was empty; manual crawls keep the
    # engine busy, so neither a completion nor spider_idle will ever fire.
    spider, enqueue = spider
    enqueue(7, ["https://a.example/1", "https://a.example/2"])

    spider._periodic_top_up()

    assert [r.url for r in scheduled(spider)] == ["https://a.example/1", "https://a.example/2"]
    assert spider._domain_pending == {7: 2}


def test_periodic_top_up_respects_low_watermark(spider):
    spider, enqueue = spider
    spider.domain_low_watermark = 1
    spider._domain_pending = {1: 5}
    enqueue(7, ["https://a.example/1"])

    spider._periodic_top_up()

    assert scheduled(spider) == []


def test_dropped_owned_request_releases_its_domain(spider):
    spider, enqueue = spider
    enqueue(7, ["https://a.example/1"])
    spider._periodic_top_up()
    (request,) = scheduled(spider)

    spider.req_dropped(request)

    assert spider._domain_pending == {}
    assert spider._pending_requests == 0


def test_dropped_request_not_owned_by_queue_is_ignored(spider):
    spider, _ = spider
    spider._domain_pending = {7: 1}
    spider._pending_requests = 1

    spider.req_dropped(scrapy.Request("https://manual.example/", meta={"manual_bulk_crawl_job_id": "j"}))

    assert spider._domain_pending == {7: 1}
    assert spider._pending_requests == 1


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    import crawler.spiders.spider as spider_module

    clock = Clock()
    monkeypatch.setattr(spider_module, "monotonic", clock)
    return clock


def test_domain_with_no_finished_request_is_reported_and_warned_once(spider, clock, caplog):
    # Production 2026-11-11: one wikipedia.org request never called back, so
    # the domain's batch stayed open and its next queue file was never taken.
    spider, enqueue = spider
    spider.domain_stall_warn_sec = 600
    enqueue(7, ["https://a.example/1", "https://a.example/2"])
    enqueue(8, ["https://b.example/1"])
    spider._periodic_top_up()
    first, _, other = scheduled(spider)

    clock.now += 300
    spider._finish_owned_request("parse", domain_id=7)
    clock.now += 400
    with caplog.at_level("INFO", logger="crawler"):
        assert spider._domain_stall_runtime() == {
            "stalled_domains": 1, "oldest_domain_idle_sec": 700, "oldest_idle_domain_id": 8,
        }
        spider._domain_stall_runtime()
    warnings = [r for r in caplog.records if getattr(r, "event", None) == "spider.domain_stalled"]
    assert [(r.domain_id, r.pending, r.idle_sec) for r in warnings] == [(8, 1, 700)]

    clock.now += 20
    with caplog.at_level("INFO", logger="crawler"):
        spider._finish_owned_request("parse", domain_id=8)
    (resumed,) = [r for r in caplog.records if getattr(r, "event", None) == "spider.domain_resumed"]
    assert (resumed.domain_id, resumed.stalled_sec, resumed.reason) == (8, 20, "parse")
    assert spider._domain_stall_runtime() == {
        "stalled_domains": 0, "oldest_domain_idle_sec": 420, "oldest_idle_domain_id": 7,
    }


def test_closed_batches_leave_no_stall_state(spider, clock):
    spider, enqueue = spider
    enqueue(7, ["https://a.example/1"])
    spider._periodic_top_up()
    spider._finish_owned_request("parse", domain_id=7)
    clock.now += 10_000
    assert spider._domain_progress == {}
    assert spider._domain_stall_runtime() == {
        "stalled_domains": 0, "oldest_domain_idle_sec": 0, "oldest_idle_domain_id": None,
    }


def test_zero_threshold_disables_the_warning_but_keeps_the_idle_age(spider, clock, caplog):
    spider, enqueue = spider
    spider.domain_stall_warn_sec = 0
    enqueue(7, ["https://a.example/1"])
    spider._periodic_top_up()
    clock.now += 10_000
    with caplog.at_level("INFO", logger="crawler"):
        runtime = spider._domain_stall_runtime()
    assert runtime["stalled_domains"] == 0
    assert runtime["oldest_domain_idle_sec"] == 10_000
    assert not [r for r in caplog.records if getattr(r, "event", None) == "spider.domain_stalled"]


def test_queue_stats_report_waiting_files_and_oldest_age(spider):
    import os
    import time

    spider, enqueue = spider
    assert spider.queue.stats() == {"queue_files": 0, "queue_oldest_age_sec": 0}
    enqueue(7, ["https://a.example/1"])
    enqueue(8, ["https://b.example/1"])
    old = time.time() - 3600
    os.utime(spider.queue.queue_dir / "domain_7.json", (old, old))
    (spider.queue.queue_dir / "unrelated.txt").write_text("x")
    stats = spider.queue.stats()
    assert stats["queue_files"] == 2
    assert 3599 <= stats["queue_oldest_age_sec"] <= 3601


def test_missing_queue_dir_reports_empty(tmp_path):
    from crawler.queue_consumer import QueueConsumer

    assert QueueConsumer(str(tmp_path / "missing")).stats() == {"queue_files": 0, "queue_oldest_age_sec": 0}


def test_heartbeat_includes_stall_and_queue_fields(spider, clock, caplog):
    spider, enqueue = spider
    enqueue(7, ["https://a.example/1"])
    spider._periodic_top_up()
    from types import SimpleNamespace
    spider.crawler.engine = SimpleNamespace(
        downloader=SimpleNamespace(slots={}, active=set()), scraper=None, _slot=None)
    clock.now += 30
    with caplog.at_level("INFO", logger="crawler"):
        spider._emit_heartbeat()
    (heartbeat,) = [r for r in caplog.records if getattr(r, "event", None) == "spider.heartbeat"]
    assert heartbeat.oldest_domain_idle_sec == 30
    assert heartbeat.oldest_idle_domain_id == 7
    assert heartbeat.stalled_domains == 0
    assert heartbeat.queue_files == 0


@pytest.mark.parametrize("value,expected", [(None, 600.0), ("0", 0.0), ("120", 120.0)])
def test_stall_threshold_comes_from_environment(monkeypatch, value, expected):
    import runpy

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.chdir(root)
    if value is None:
        monkeypatch.delenv("CRAWLER_DOMAIN_STALL_WARN_SEC", raising=False)
    else:
        monkeypatch.setenv("CRAWLER_DOMAIN_STALL_WARN_SEC", value)
    assert runpy.run_module("containers.crawler.crawler.settings")["DOMAIN_STALL_WARN_SEC"] == expected
