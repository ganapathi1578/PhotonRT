from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic_ns, perf_counter
from typing import Iterator, Optional, Union

from . import _photonrt

PathLike = Union[str, Path]


@dataclass(frozen=True)
class CaptionResult:
    frame_id: int
    capture_time_ns: int
    caption: str
    queue_wait_ms: float
    inference_ms: float
    total_latency_ms: float


@dataclass
class _FrameTask:
    frame_id: int
    capture_time_ns: int
    frame: object


class CameraCaptioner:
    """
    Parallel camera-stream captioning built on the existing native PhotonRT
    inference runtime.

    v1 uses OpenCV only for camera capture. The hot inference path remains in
    C++: BGR->RGB, PhotonRT preprocessing, MobileCLIP FP32, Photon prefill,
    Photon decode, and tokenization.

    Every submitted frame gets a frame_id and capture timestamp. Each result
    returns those identifiers so a downstream display/playback layer can
    synchronize the caption to the original frame.
    """

    def __init__(
        self,
        source,
        mobileclip_model: PathLike,
        photon_prefill_model: PathLike,
        photon_decode_model: PathLike,
        tokenizer: PathLike,
        *,
        workers: int = 2,
        queue_size: int = 8,
        frame_stride: int = 1,
        max_new_tokens: int = 32,
        drop_oldest: bool = True,
        capture_api: Optional[int] = None,
    ):
        if workers <= 0:
            raise ValueError("workers must be > 0")
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

        self.source = source
        self.mobileclip_model = str(mobileclip_model)
        self.photon_prefill_model = str(photon_prefill_model)
        self.photon_decode_model = str(photon_decode_model)
        self.tokenizer = str(tokenizer)

        self.workers = int(workers)
        self.queue_size = int(queue_size)
        self.frame_stride = int(frame_stride)
        self.max_new_tokens = int(max_new_tokens)
        self.drop_oldest = bool(drop_oldest)
        self.capture_api = capture_api

        self._tasks: Queue[_FrameTask] = Queue(maxsize=self.queue_size)
        self._results: Queue[CaptionResult] = Queue(
            maxsize=max(2, self.queue_size * 2)
        )

        self._stop = Event()
        self._started = False
        self._start_lock = Lock()

        self._capture_thread: Optional[Thread] = None
        self._worker_threads: list[Thread] = []

        self._error: Optional[BaseException] = None
        self._error_lock = Lock()

        self._captured_frames = 0
        self._submitted_frames = 0
        self._completed_frames = 0
        self._dropped_frames = 0

    def start(self) -> "CameraCaptioner":
        import cv2

        with self._start_lock:
            if self._started:
                return self

            self._stop.clear()
            with self._error_lock:
                self._error = None

            if self.capture_api is None:
                cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source, self.capture_api)

            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    f"Could not open camera/video source: {self.source!r}"
                )

            self._capture_thread = Thread(
                target=self._capture_loop,
                args=(cap,),
                name="photonrt-camera-capture",
                daemon=True,
            )
            self._capture_thread.start()

            for worker_id in range(self.workers):
                thread = Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    name=f"photonrt-camera-worker-{worker_id}",
                    daemon=True,
                )
                thread.start()
                self._worker_threads.append(thread)

            self._started = True

        return self

    def stop(self) -> None:
        with self._start_lock:
            if not self._started:
                return

            self._stop.set()

            if self._capture_thread is not None:
                self._capture_thread.join(timeout=2.0)

            for thread in self._worker_threads:
                thread.join(timeout=10.0)

            self._worker_threads.clear()
            self._capture_thread = None
            self._started = False

    def __enter__(self) -> "CameraCaptioner":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __iter__(self) -> Iterator[CaptionResult]:
        self.start()

        while not self._stop.is_set():
            try:
                result = self._results.get(timeout=0.25)
            except Empty:
                self._raise_error()
                continue

            self._completed_frames += 1
            yield result

        self._raise_error()

    def get_result(self, timeout: Optional[float] = None) -> CaptionResult:
        self.start()
        self._raise_error()

        try:
            result = self._results.get(timeout=timeout)
        except Empty:
            self._raise_error()
            raise

        self._completed_frames += 1
        return result

    def stats(self) -> dict[str, int]:
        return {
            "captured_frames": self._captured_frames,
            "submitted_frames": self._submitted_frames,
            "completed_frames": self._completed_frames,
            "dropped_frames": self._dropped_frames,
            "queued_frames": self._tasks.qsize(),
        }

    def _capture_loop(self, cap) -> None:
        frame_id = 0

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._stop.set()
                    break

                capture_time_ns = monotonic_ns()
                self._captured_frames += 1

                current_id = frame_id
                frame_id += 1

                if current_id % self.frame_stride != 0:
                    continue

                self._put_task(
                    _FrameTask(
                        frame_id=current_id,
                        capture_time_ns=capture_time_ns,
                        frame=frame,
                    )
                )

        except BaseException as exc:
            self._set_error(exc)
            self._stop.set()
        finally:
            cap.release()

    def _put_task(self, task: _FrameTask) -> None:
        try:
            self._tasks.put_nowait(task)
            self._submitted_frames += 1
            return
        except Full:
            pass

        if not self.drop_oldest:
            self._dropped_frames += 1
            return

        try:
            old = self._tasks.get_nowait()
            del old
            self._tasks.task_done()
            self._dropped_frames += 1
        except Empty:
            pass

        try:
            self._tasks.put_nowait(task)
            self._submitted_frames += 1
        except Full:
            self._dropped_frames += 1

    def _worker_loop(self, worker_id: int) -> None:
        # Each worker owns its own native runtime.
        runtime = _photonrt.Captioner(
            self.mobileclip_model,
            self.photon_prefill_model,
            self.photon_decode_model,
            self.tokenizer,
        )

        options = _photonrt.CaptionOptions()
        options.max_new_tokens = self.max_new_tokens

        while not self._stop.is_set():
            try:
                task = self._tasks.get(timeout=0.25)
            except Empty:
                continue

            try:
                # ------------------------------------------------
                # Queue wait
                # ------------------------------------------------
                worker_start_ns = monotonic_ns()

                queue_wait_ms = (
                    worker_start_ns - task.capture_time_ns
                ) / 1_000_000.0

                # ------------------------------------------------
                # Native inference
                # ------------------------------------------------
                t0 = perf_counter()

                # OpenCV frame:
                # BGR uint8 HWC
                #
                # Native C++ handles:
                # BGR -> RGB
                # preprocessing
                # MobileCLIP
                # Photon prefill
                # Photon decode
                # tokenizer
                caption = runtime.caption_bgr(
                    task.frame,
                    options,
                )

                inference_ms = (
                    perf_counter() - t0
                ) * 1000.0

                # ------------------------------------------------
                # End-to-end latency
                # ------------------------------------------------
                total_latency_ms = (
                    monotonic_ns() - task.capture_time_ns
                ) / 1_000_000.0

                self._put_result(
                    CaptionResult(
                        frame_id=task.frame_id,
                        capture_time_ns=task.capture_time_ns,
                        caption=caption,
                        queue_wait_ms=queue_wait_ms,
                        inference_ms=inference_ms,
                        total_latency_ms=total_latency_ms,
                    )
                )

            except BaseException as exc:
                self._set_error(exc)
                self._stop.set()

            finally:
                self._tasks.task_done()
    def _put_result(self, result: CaptionResult) -> None:
        try:
            self._results.put_nowait(result)
            return
        except Full:
            pass

        # Keep the newest result when a slow consumer falls behind.
        try:
            self._results.get_nowait()
        except Empty:
            pass

        try:
            self._results.put_nowait(result)
        except Full:
            pass

    def _set_error(self, exc: BaseException) -> None:
        with self._error_lock:
            self._error = exc

    def _raise_error(self) -> None:
        with self._error_lock:
            error = self._error

        if error is not None:
            raise RuntimeError("PhotonRT camera pipeline failed") from error

    @classmethod
    def from_pretrained(
        cls,
        camera=0,
        repo_id="ganapathi1578/PhotonRT",
        revision="main",
        cache_dir=None,
        workers=1,
        frame_stride=15,
        **kwargs,
    ):
        """
        Create a CameraCaptioner using a Hugging Face PhotonRT model.
        """

        from .hub import download_model

        paths = download_model(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
        )

        return cls(
            camera,
            str(paths["mobileclip"]),
            str(paths["prefill"]),
            str(paths["decode"]),
            str(paths["tokenizer"]),
            workers=workers,
            frame_stride=frame_stride,
            **kwargs,
        )
