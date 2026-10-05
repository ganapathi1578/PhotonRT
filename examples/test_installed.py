# from photonrt import Captioner

# model = Captioner.from_pretrained(
#     "ganapathi1578/photonrt"
# )

# print(
#     model.caption(
#         r"C:\Users\GANAPATHI\Downloads\687474703a2f2f696d616765732e636f636f646174617365742e6f7267\val2017\000000245915.jpg"
#     )
# )


# from photonrt import CameraCaptioner

# camera = CameraCaptioner(
#     0,
#     "models/fp32/mobileclip-s1.onnx",
#     "models/fp32/photon_prefill.onnx",
#     "models/fp32/photon.onnx",
#     "models/fp32/photon.tokenizer",
#     workers=2,
#     frame_stride=5,
# )

# try:
#     for result in camera:
#         print(
#             f"frame={result.frame_id} "
#             f"latency={result.total_latency_ms:.1f} ms "
#             f"{result.caption}"
#         )
# finally:
#     camera.stop()




import cv2

from photonrt import Captioner, CaptionOptions

model = Captioner(
    "models/fp32/mobileclip-s1.onnx",
    "models/fp32/photon_prefill.onnx",
    "models/fp32/photon.onnx",
    "models/fp32/photon.tokenizer",
)

image = cv2.imread(r"C:\Users\GANAPATHI\Downloads\687474703a2f2f696d616765732e636f636f646174617365742e6f72672f76616c323031372f3030303030303234353931352e6a7067.jpg"
)

if image is None:
    raise RuntimeError("Failed to load image")

options = CaptionOptions()
options.max_new_tokens = 32

caption = model._native.caption_bgr(
    image,
    options,
)

print("shape:", image.shape)
print("dtype:", image.dtype)
print("caption:", caption)