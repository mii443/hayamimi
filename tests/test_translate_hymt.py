"""Model-free tests for trilingual Hy-MT2 routing and event association."""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from realtime_transcribe import TranslationWorker, safe_translate_pair
from subtitle_server import SubtitleServer, _dashboard_html
from translate_hymt import HyMTClient, llama_server_command, translation_targets


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.body = body

    def read(self):
        return json.dumps(self.body, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_targets_are_the_other_two_languages_in_display_order():
    assert translation_targets("ja") == ("en", "ko")
    assert translation_targets("en") == ("ja", "ko")
    assert translation_targets("ko") == ("ja", "en")
    assert translation_targets("zh") == ()


def test_client_sends_explicit_source_and_target_prompt():
    requests = []

    def opener(request, timeout):
        requests.append((request, timeout))
        return FakeResponse({"choices": [{"message": {"content": "日本語訳"}}]})

    client = HyMTClient("http://translation:18081", timeout=7.5, opener=opener)
    assert client.translate("An update is ready.", "en", "ja") == "日本語訳"

    request, timeout = requests[0]
    payload = json.loads(request.data.decode("utf-8"))
    prompt = payload["messages"][0]["content"]
    assert request.full_url == "http://translation:18081/v1/chat/completions"
    assert timeout == 7.5
    assert "English text into Japanese" in prompt
    assert "Preserve every Arabic digit sequence exactly as written" in prompt
    assert prompt.endswith("An update is ready.")
    assert payload["top_k"] == 20


def test_client_rejects_invalid_or_same_language_pairs():
    client = HyMTClient(opener=lambda *_args, **_kwargs: None)
    for source, target in (("zh", "ja"), ("ja", "zh"), ("ko", "ko")):
        try:
            client.translate("text", source, target)
        except ValueError:
            pass
        else:
            raise AssertionError(f"pair should have failed: {source}->{target}")


def test_client_limits_requests_shared_by_live_and_refiner_callers():
    lock = threading.Lock()
    active = 0
    maximum = 0

    def opener(_request, timeout):
        assert timeout == 10.0
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return FakeResponse({"choices": [{"message": {"content": "translated"}}]})

    client = HyMTClient(opener=opener, max_concurrency=1)
    threads = [
        threading.Thread(target=client.translate, args=("text", "en", "ja"))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert maximum == 1


def test_server_command_supports_classic_and_unified_llama_binaries():
    classic = llama_server_command("/usr/bin/llama-server", "/m.gguf",
                                   "127.0.0.1", 18081, 2)
    unified = llama_server_command("/opt/llama", "/m.gguf",
                                   "127.0.0.1", 18081, 2)
    assert classic[:3] == ["/usr/bin/llama-server", "-m", "/m.gguf"]
    assert unified[:4] == ["/opt/llama", "serve", "-m", "/m.gguf"]
    assert ["-ngl", "99"] == classic[3:5]


class FakeHyMT:
    def targets_for(self, source_lang):
        return translation_targets(source_lang)

    def translate(self, text, source_lang, target_lang):
        return f"{source_lang}->{target_lang}:{text}"


class EventSink:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


def test_worker_emits_two_keyed_translations_for_each_trilingual_final():
    sink = EventSink()
    worker = TranslationWorker(FakeHyMT(), server=sink, workers=2)
    worker.submit("hello", "en", utterance_id="7-3")
    worker.close(wait=True)

    assert {(event["source_lang"], event["lang"]) for event in sink.events} == {
        ("en", "ja"), ("en", "ko")}
    assert {event["utterance_id"] for event in sink.events} == {"7-3"}
    assert all(event["type"] == "translation" for event in sink.events)


def test_worker_ignores_languages_outside_the_trilingual_set():
    sink = EventSink()
    worker = TranslationWorker(FakeHyMT(), server=sink)
    worker.submit("你好", "zh", utterance_id="2-1")
    worker.close(wait=True)
    assert sink.events == []


def test_pair_translation_keeps_source_when_digits_change():
    class BadNumbers:
        calls = 0

        def translate(self, _text, _source, _target):
            self.calls += 1
            return "It costs 5 yen."

    translator = BadNumbers()
    assert safe_translate_pair(translator, "500円です。", "ja", "en") == "500円です。"
    assert translator.calls == 2


def test_pair_translation_requires_every_repeated_source_number():
    class LosesRepeatedNumber:
        def translate(self, _text, _source, _target):
            return "500円です。"

    assert safe_translate_pair(
        LosesRepeatedNumber(), "500원과 500원입니다.", "ko", "ja"
    ) == "500원과 500원입니다."


def test_pair_translation_retries_a_transient_failure():
    class TransientFailure:
        calls = 0

        def translate(self, _text, _source, _target):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            return "日本語訳です。"

    translator = TransientFailure()
    assert safe_translate_pair(translator, "한국어입니다.", "ko", "ja") == "日本語訳です。"
    assert translator.calls == 2


def test_pair_translation_retries_source_text_response():
    class SourceThenTranslation:
        calls = 0

        def translate(self, text, _source, _target):
            self.calls += 1
            return text if self.calls == 1 else "日本語訳です。"

    translator = SourceThenTranslation()
    assert safe_translate_pair(translator, "한국어입니다.", "ko", "ja") == "日本語訳です。"
    assert translator.calls == 2


def test_pair_translation_retries_changed_digit_rendering():
    class KanjiDigitsThenPreserved:
        calls = 0

        def translate(self, _text, _source, _target):
            self.calls += 1
            if self.calls == 1:
                return "千九百四十年に始まりました。"
            return "1940年に始まりました。"

    translator = KanjiDigitsThenPreserved()
    assert safe_translate_pair(
        translator, "1940년에 시작했습니다.", "ko", "ja"
    ) == "1940年に始まりました。"
    assert translator.calls == 2


def test_pair_translation_keeps_source_after_two_exceptions():
    class AlwaysFails:
        calls = 0

        def translate(self, _text, _source, _target):
            self.calls += 1
            raise TimeoutError("still unavailable")

    translator = AlwaysFails()
    assert safe_translate_pair(translator, "한국어입니다.", "ko", "ja") == "한국어입니다."
    assert translator.calls == 2


def test_worker_publishes_japanese_after_retry_and_then_english():
    class JapaneseRetry:
        def __init__(self):
            self.calls = {}

        def targets_for(self, source_lang):
            return translation_targets(source_lang)

        def translate(self, text, _source, target):
            self.calls[target] = self.calls.get(target, 0) + 1
            if target == "ja" and self.calls[target] == 1:
                return text
            return {"ja": "日本語訳です。", "en": "English translation."}[target]

    sink = EventSink()
    translator = JapaneseRetry()
    worker = TranslationWorker(translator, server=sink, workers=1)
    worker.submit("한국어입니다.", "ko", utterance_id="4-2")
    worker.close(wait=True)

    assert [(event["lang"], event["text"]) for event in sink.events] == [
        ("ja", "日本語訳です。"),
        ("en", "English translation."),
    ]
    assert translator.calls == {"ja": 2, "en": 1}
    assert worker.status()["failed_targets"] == 0


def test_worker_continues_after_one_target_publish_failure():
    class FailsFirstPublish(EventSink):
        def publish(self, event):
            if not self.events:
                self.events.append(None)
                raise RuntimeError("temporary sink failure")
            self.events.append(event)

    sink = FailsFirstPublish()
    worker = TranslationWorker(FakeHyMT(), server=sink, workers=1)
    worker.submit("hello", "en", utterance_id="8-1")
    worker.close(wait=True)

    assert sink.events[0] is None
    assert [event["lang"] for event in sink.events[1:]] == ["ko"]
    assert worker.status()["failed_targets"] == 1


def test_worker_queue_accepts_or_drops_both_target_languages_together():
    class BlockingHyMT(FakeHyMT):
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()

        def translate(self, text, source_lang, target_lang):
            if not self.entered.is_set():
                self.entered.set()
                assert self.release.wait(timeout=2)
            return super().translate(text, source_lang, target_lang)

    sink = EventSink()
    translator = BlockingHyMT()
    worker = TranslationWorker(
        translator, server=sink, workers=1, queue_capacity=2
    )
    worker.submit("one", "ko", utterance_id="1")
    assert translator.entered.wait(timeout=1)
    worker.submit("two", "ko", utterance_id="2")
    worker.submit("three", "ko", utterance_id="3")
    worker.submit("four", "ko", utterance_id="4")
    assert worker.status()["dropped_batches"] == 1

    translator.release.set()
    worker.close(wait=True)

    by_utterance = {}
    for event in sink.events:
        by_utterance.setdefault(event["utterance_id"], set()).add(event["lang"])
    assert by_utterance == {
        "1": {"ja", "en"},
        "2": {"ja", "en"},
        "3": {"ja", "en"},
    }


def test_submit_cannot_put_a_batch_behind_close_stop_markers():
    sink = EventSink()
    worker = TranslationWorker(FakeHyMT(), server=sink, workers=1)
    original_put = worker._q.put_nowait
    put_entered = threading.Event()
    allow_put = threading.Event()

    def controlled_put(item):
        if isinstance(item, tuple) and len(item) == 5:
            put_entered.set()
            assert allow_put.wait(timeout=2)
        original_put(item)

    worker._q.put_nowait = controlled_put
    submitter = threading.Thread(
        target=worker.submit, args=("hello", "en", "race", sink)
    )
    submitter.start()
    assert put_entered.wait(timeout=1)
    closer = threading.Thread(target=worker.close)
    closer.start()
    time.sleep(0.02)
    assert closer.is_alive()
    allow_put.set()
    submitter.join(timeout=2)
    closer.join(timeout=2)

    assert not submitter.is_alive()
    assert not closer.is_alive()
    assert {event["lang"] for event in sink.events} == {"ja", "ko"}
    assert worker._q.unfinished_tasks == 0


def test_worker_close_has_one_total_deadline():
    class BlockingHyMT(FakeHyMT):
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()

        def translate(self, text, source_lang, target_lang):
            self.entered.set()
            assert self.release.wait(timeout=2)
            return super().translate(text, source_lang, target_lang)

    translator = BlockingHyMT()
    worker = TranslationWorker(translator, workers=1)
    worker.submit("hello", "en", utterance_id="deadline")
    assert translator.entered.wait(timeout=1)
    started = time.monotonic()
    worker.close(wait=True, timeout=0.05)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5

    translator.release.set()
    for thread in worker._threads:
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_local_final_returns_id_and_publishes_it():
    server = SubtitleServer()
    queue = server.subscribe()
    utterance_id = server.final("hello", "en")
    event = json.loads(queue.get_nowait())
    server.unsubscribe(queue)
    assert utterance_id == "local-1"
    assert event["utterance_id"] == utterance_id


def test_dashboard_places_translation_by_stream_and_utterance():
    html = _dashboard_html()
    assert "cards.set(eventKey(ev), card)" in html
    assert "const targetCard = cards.get(eventKey(ev)) || lastCard" in html
    assert "targetCard.appendChild(tr)" in html
