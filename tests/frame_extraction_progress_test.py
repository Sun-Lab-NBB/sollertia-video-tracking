"""Contains tests for the aggregate frame-extraction progress bar and DeepLabCut tqdm shim used to report progress."""

import io
import queue

from sollertia_video_tracking.frame_extraction.progress import (
    AggregateBar,
    make_progress_reporter,
)


def _drain(progress_queue: queue.Queue) -> list:
    """Collects every message currently sitting on the queue into a list."""
    messages = []
    while not progress_queue.empty():
        messages.append(progress_queue.get_nowait())
    return messages


class _RaisingQueue:
    """A stand-in queue whose ``put_nowait`` records the attempt then always raises, exercising exception suppression.

    Recording each attempted item lets a test prove that both suppress paths (the leading-zero announce and the
    per-frame put) were genuinely reached rather than skipped, since a raise that escaped would abort iteration.
    """

    def __init__(self) -> None:
        self.attempts: list[tuple[str, int, int]] = []

    def put_nowait(self, item: tuple[str, int, int]) -> None:
        self.attempts.append(item)
        raise RuntimeError


def _make_bar(
    frame_totals: dict[int, int],
    total_video_count: int | None = None,
    stream: io.StringIO | None = None,
) -> AggregateBar:
    """Builds an AggregateBar over a throwaway queue and an in-memory (non-tty) stream."""
    if total_video_count is None:
        total_video_count = len(frame_totals)
    return AggregateBar(
        progress_queue=queue.Queue(),
        total_video_count=total_video_count,
        frame_totals=frame_totals,
        stream=stream if stream is not None else io.StringIO(),
    )


def test_reporter_forwards_items_unchanged_and_emits_every_frame_for_small_video() -> None:
    """Verifies that a tiny video's stride of one reports every frame plus the leading zero and forwards items."""
    progress_queue: queue.Queue = queue.Queue()
    # frame_total // 100 == 0, so the stride clamps up to 1: every count crosses the stride.
    reporter = make_progress_reporter(progress_queue, video_index=2, frame_total=5)

    forwarded = list(reporter(range(5)))

    assert forwarded == [0, 1, 2, 3, 4]
    # A leading zero announces the video, then one message per decoded frame (1..5, with 5 == frame_total).
    assert _drain(progress_queue) == [
        ("progress", 2, 0),
        ("progress", 2, 1),
        ("progress", 2, 2),
        ("progress", 2, 3),
        ("progress", 2, 4),
        ("progress", 2, 5),
    ]


def test_reporter_throttles_updates_by_stride() -> None:
    """Verifies that a larger frame total widens the stride, reporting only counts on it plus the lead zero."""
    progress_queue: queue.Queue = queue.Queue()
    # frame_total // 100 == 3, below the 250 cap, so the stride is 3.
    reporter = make_progress_reporter(progress_queue, video_index=0, frame_total=300)

    forwarded = list(reporter(range(10)))

    assert forwarded == list(range(10))
    # Lead zero, then only counts divisible by three; 300 (frame_total) is never reached over ten items.
    assert _drain(progress_queue) == [
        ("progress", 0, 0),
        ("progress", 0, 3),
        ("progress", 0, 6),
        ("progress", 0, 9),
    ]


def test_reporter_stride_is_capped_at_max_frames_per_update() -> None:
    """Verifies that a huge frame total does not widen the stride past the 250-frame cap."""
    progress_queue: queue.Queue = queue.Queue()
    # frame_total // 100 == 1000, but the cap holds the stride at 250.
    reporter = make_progress_reporter(progress_queue, video_index=1, frame_total=100_000)

    list(reporter(range(500)))

    # Lead zero, then only the two multiples of the capped stride reached within 500 frames.
    assert _drain(progress_queue) == [
        ("progress", 1, 0),
        ("progress", 1, 250),
        ("progress", 1, 500),
    ]


def test_reporter_emits_on_final_frame_off_the_stride() -> None:
    """Verifies that the terminal frame is reported even when the frame total is not a multiple of the stride."""
    progress_queue: queue.Queue = queue.Queue()
    # frame_total // 100 == 2 -> stride 2; 201 is odd, so only the frame-total equality triggers the last message.
    reporter = make_progress_reporter(progress_queue, video_index=4, frame_total=201)

    list(reporter(range(201)))

    messages = _drain(progress_queue)
    # 200 is the last count that lands on the stride of two; 201 is odd, so only the count == frame_total equality can
    # emit it. Removing that equality branch would leave 200 as the final message, so both asserts pin the branch.
    assert messages[-2] == ("progress", 4, 200)
    assert messages[-1] == ("progress", 4, 201)
    # The terminal frame is emitted exactly once, through the equality rather than a second stride hit.
    assert messages.count(("progress", 4, 201)) == 1


def test_reporter_suppresses_queue_errors_and_still_forwards_items() -> None:
    """Verifies that a queue that rejects every put is swallowed, so the wrapped iteration still yields every item."""
    raising_queue = _RaisingQueue()
    reporter = make_progress_reporter(raising_queue, video_index=0, frame_total=3)

    # Both the leading-zero put and the per-frame puts raise, but suppression keeps the generator forwarding items.
    forwarded = list(reporter(["a", "b", "c"]))

    assert forwarded == ["a", "b", "c"]
    # Every put was actually attempted (and each raised): the leading-zero announce plus one per frame at stride 1.
    # This confirms both contextlib.suppress paths were exercised, not merely that no exception escaped.
    assert raising_queue.attempts == [
        ("progress", 0, 0),
        ("progress", 0, 1),
        ("progress", 0, 2),
        ("progress", 0, 3),
    ]


def test_reporter_accepts_and_ignores_extra_tqdm_arguments() -> None:
    """Verifies that the reporter, a tqdm drop-in, tolerates extra positional and keyword arguments tqdm receives."""
    progress_queue: queue.Queue = queue.Queue()
    reporter = make_progress_reporter(progress_queue, video_index=0, frame_total=3)

    # This library rebinds ``dlc_videos.tqdm`` to the reporter on the inference path and passes it as ``progress=`` on
    # the frame-extraction path, and both call it with a single positional iterable today. The extra *args/**kwargs
    # keep the shim a safe drop-in should a call site start passing tqdm's desc, total, or leave.
    forwarded = list(reporter(range(3), "frames", total=3, leave=False))

    assert forwarded == [0, 1, 2]
    assert _drain(progress_queue) == [
        ("progress", 0, 0),
        ("progress", 0, 1),
        ("progress", 0, 2),
        ("progress", 0, 3),
    ]


def test_aggregate_bar_grand_total_clamped_to_one_for_empty_totals() -> None:
    """Verifies that an empty frame-total mapping clamps the grand total to one, avoiding a fraction divide-by-zero."""
    bar = _make_bar({}, total_video_count=0)

    assert bar._grand_frame_total == 1
    assert bar._frames == {}
    assert bar._videos_done == 0


def test_aggregate_bar_grand_total_sums_per_video_totals() -> None:
    """Verifies that the grand total is the sum of the per-video frame totals when the mapping is non-empty."""
    bar = _make_bar({0: 100, 1: 250})

    assert bar._grand_frame_total == 350
    assert bar._total_video_count == 2


def test_ingest_progress_updates_frame_count_without_forcing_redraw() -> None:
    """Verifies that a progress message stores the latest count for its video and does not force an immediate redraw."""
    bar = _make_bar({3: 100})

    force = bar._ingest(("progress", 3, 42))

    assert force is False
    assert bar._frames == {3: 42}
    assert bar._videos_done == 0


def test_ingest_done_completes_known_video_and_forces_redraw() -> None:
    """Verifies that a done message fills the video to its full total, counts it done, and forces a redraw."""
    bar = _make_bar({1: 100})

    force = bar._ingest(("done", 1))

    assert force is True
    assert bar._frames[1] == 100
    assert bar._videos_done == 1


def test_ingest_done_for_unknown_video_defaults_to_zero() -> None:
    """Verifies that a done message for a video absent from the totals defaults its frame count to zero."""
    bar = _make_bar({1: 100})

    force = bar._ingest(("done", 7))

    assert force is True
    assert bar._frames[7] == 0
    assert bar._videos_done == 1


def test_ingest_unknown_message_kind_is_ignored() -> None:
    """Verifies that a message that is neither progress nor done leaves the state untouched and forces no redraw."""
    bar = _make_bar({1: 100})

    force = bar._ingest(("mystery", 9))

    assert force is False
    assert bar._frames == {}
    assert bar._videos_done == 0


def test_is_preparing_true_before_any_work() -> None:
    """Verifies that a freshly constructed bar reports it is preparing, since no frames or completions have arrived."""
    bar = _make_bar({0: 100})

    assert bar._is_preparing() is True


def test_is_preparing_false_after_first_progress() -> None:
    """Verifies that once any frame count arrives, the bar is no longer preparing."""
    bar = _make_bar({0: 100})
    bar._ingest(("progress", 0, 5))

    assert bar._is_preparing() is False


def test_is_preparing_false_after_completion() -> None:
    """Verifies that a completed video also takes the bar out of the preparing state."""
    bar = _make_bar({0: 100})
    bar._ingest(("done", 0))

    assert bar._is_preparing() is False


def test_compose_preparing_shows_label_video_count_and_elapsed() -> None:
    """Verifies that the warm-up line carries the label, the queued video count, and the elapsed clock."""
    bar = _make_bar({0: 100, 1: 100}, total_video_count=2)

    line = bar._compose_preparing(elapsed=90.0)

    assert line.startswith(f"[{'-' * bar._width}] ")
    assert "preparing..." in line
    assert "0/2 videos" in line
    assert "01:30 elapsed" in line


def test_compose_active_reports_fraction_active_count_and_frames() -> None:
    """Verifies that the active line renders the completion percent, the decoding count, and the frames-read tally."""
    bar = _make_bar({0: 100, 1: 100}, total_video_count=2)
    bar._ingest(("progress", 0, 50))
    bar._ingest(("progress", 1, 30))

    line = bar._compose_active(elapsed=10.0)

    # 80 of 200 frames read -> 40 percent; two videos have announced and none are done, so both are decoding.
    assert " 40.0% " in line
    assert "0/2 videos" in line
    assert "(2 decoding)" in line
    assert "80/200 frames" in line


def test_compose_active_clamps_frames_and_zero_active_when_all_done() -> None:
    """Verifies that over-reported frames clamp to the grand total, and a done video drops from the decoding count."""
    bar = _make_bar({0: 100}, total_video_count=1)
    # First an over-report, then completion: the sum would exceed the grand total but is clamped, and the finished
    # video drops out of the decoding count.
    bar._ingest(("progress", 0, 150))
    bar._ingest(("done", 0))

    line = bar._compose_active(elapsed=10.0)

    assert "100.0%" in line
    assert "1/1 videos" in line
    assert "(0 decoding)" in line
    assert "100/100 frames" in line
    # A finished run projects a zero ETA rather than a running estimate.
    assert "ETA 00:00" in line
