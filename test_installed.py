from photonrt import Captioner

model = Captioner.from_pretrained(
    "ganapathi1578/photonrt"
)

print(
    model.caption(
        r"C:\Users\GANAPATHI\Downloads\687474703a2f2f696d616765732e636f636f646174617365742e6f7267\val2017\000000245915.jpg"
    )
)