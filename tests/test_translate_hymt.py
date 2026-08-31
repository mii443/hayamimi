"""Model-free tests for trilingual Hy-MT2 routing and event association."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from realtime_transcribe import TranslationWorker, safe_translate_pair
from subtitle_server import SubtitleServer, _dashboard_html
from translate_hymt import HyMTClient, translation_targets


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
        def translate(self, _text, _source, _target):
            return "It costs 5 yen."

    assert safe_translate_pair(BadNumbers(), "500円です。", "ja", "en") == "500円です。"


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
