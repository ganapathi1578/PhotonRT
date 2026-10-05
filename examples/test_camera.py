# from photonrt import CameraCaptioner


# camera = CameraCaptioner(
#     0,
#     "models/fp32/mobileclip-s1.onnx",
#     "models/fp32/photon_prefill.onnx",
#     "models/fp32/photon.onnx",
#     "models/fp32/photon.tokenizer",
#     workers=1,
#     frame_stride=10,
# )

# try:
#     for result in camera:
#         print(
#             f"frame={result.frame_id} "
#             f"latency={result.total_latency_ms:.1f} ms "
#             f"{result.caption}"
#         )

# except KeyboardInterrupt:
#     print("\nStopping camera...")

# finally:
#     camera.stop()

from photonrt import CameraCaptioner


camera = CameraCaptioner.from_pretrained(
    # camera=0,
    # workers=1,
    # frame_stride=10,
)

try:
    for result in camera:
        print(
            f"frame={result.frame_id} "
            f"queue={result.queue_wait_ms:.1f} ms "
            f"infer={result.inference_ms:.1f} ms "
            f"total={result.total_latency_ms:.1f} ms "
            f"{result.caption}"
        )

except KeyboardInterrupt:
    print("\nStopping camera...")

finally:
    camera.stop()